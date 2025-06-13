"""エージェントの基底クラスと認知アーキテクチャ対応エージェントを定義するモジュール."""

from __future__ import annotations

import os
import random
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING
from datetime import UTC, datetime
from threading import Thread

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk
from aiwolf_nlp_common.role import Role
from aiwolf_nlp_common.status import Status

from utils.agent_logger import AgentLogger
from utils.game_logger import GameLogger
from utils.stoppable_thread import StoppableThread

from cognitive.belief_generator import BeliefGenerator
from cognitive.desire_generator import DesireGenerator
from cognitive.intention_generator import IntentionGenerator
from cognitive.speech_generator import SpeechGenerator
from cognitive.model_types import Belief, Desire, Intention

if TYPE_CHECKING:
    from collections.abc import Callable


class StoppableThread(Thread):
    """停止可能なスレッド."""

    def __init__(self, *args, **kwargs):
        """スレッドを初期化する."""
        super().__init__(*args, **kwargs)
        self._stop_event = False

    def stop(self):
        """スレッドを停止する."""
        self._stop_event = True

    def stopped(self):
        """スレッドが停止しているかどうかを返す."""
        return self._stop_event


class Agent:
    """エージェントの基底クラス."""

    def __init__(
        self,
        config: dict,
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.game_logger = GameLogger(config, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []

        # 認知モデル
        self.belief_generator: BeliefGenerator | None = None
        self.desire_generator: DesireGenerator | None = None
        self.intention_generator: IntentionGenerator | None = None
        self.speech_generator: SpeechGenerator | None = None

        self.status_map: dict[str, Status] = {}  # status_mapをここで初期化

        self.beliefs: list[Belief] = []
        self.desires: list[Desire] = []
        self.intentions: list[Intention] = []

        load_dotenv(Path(__file__).parent.joinpath("./../config/.env"))

    @staticmethod
    def timeout(func: Callable) -> Callable:
        def _wrapper(self, *args, **kwargs) -> str:
            res = ""

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(self, *args, **kwargs)
                except Exception as e:
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            timeout_value = (
                self.setting.timeout.action
                if hasattr(self, "setting") and self.setting
                else 0
            ) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning("アクションがタイムアウトしました: %s", self.request)
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning("アクションを強制終了しました: %s", self.request)
            else:
                thread.join()
            if isinstance(res, Exception):
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history = []
            self.whisper_history = []
            self.llm_message_history = []
        self.agent_logger.logger.debug(packet)

    def get_alive_agents(self) -> list[str]:
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        if request is None:
            return None
        if request.lower() not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request.lower()]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
        }
        template: Template = Template(prompt)
        prompt = template.render(**key).strip()
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        try:
            self.llm_message_history.append(HumanMessage(content=prompt))
            response = self.llm_model.invoke(self.llm_message_history)
            response_content = (
                response.content if isinstance(response.content, str) else str(response.content[0])
            )
            self.llm_message_history.append(AIMessage(content=response_content))
            self.agent_logger.logger.info(["LLM", prompt, response_content])
        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None
        else:
            return response_content

    @timeout
    def name(self) -> str:
        return self.agent_name

    def initialize(self) -> None:
        if self.config is None or self.info is None:
            return
        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
        self._send_message_to_llm(self.request)

    def daily_initialize(self) -> None:
        self._send_message_to_llm(self.request)

    def day_start(self) -> None:
        self.belief_generator = BeliefGenerator(self)
        self.desire_generator = DesireGenerator(self)
        self.intention_generator = IntentionGenerator(self)
        self.speech_generator = SpeechGenerator(self)

        try:
            self.beliefs = self.belief_generator.generate_beliefs()
        except Exception as e:
            self.agent_logger.logger.error(f"Belief generation failed: {e}")
            self.beliefs = []

        try:
            self.desires = self.desire_generator.generate_desires(self.beliefs)
        except Exception as e:
            self.agent_logger.logger.error(f"Desire generation failed: {e}")
            self.desires = []

        try:
            self.intentions = self.intention_generator.generate_intentions(self.beliefs, self.desires)
        except Exception as e:
            self.agent_logger.logger.error(f"Intention generation failed: {e}")
            self.intentions = []

    def talk(self) -> str:
        if self.speech_generator is None:
            self.speech_generator = SpeechGenerator(self)
        try:
            return self.speech_generator.generate_speech(self.beliefs, self.desires, self.intentions)
        except Exception as e:
            self.agent_logger.logger.error(f"Speech generation failed: {e}")
            return ""

    def vote(self) -> str | int:
        for intention in self.intentions:
            if intention.action_type.lower() == "vote" and intention.target_agent is not None:
                return intention.target_agent
        return random.choice(self.get_alive_agents())

    def whisper(self) -> str:
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        return response or ""

    def daily_finish(self) -> None:
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        return self._send_message_to_llm(self.request) or random.choice(self.get_alive_agents())

    def guard(self) -> str:
        return self._send_message_to_llm(self.request) or random.choice(self.get_alive_agents())

    def attack(self) -> str:
        return self._send_message_to_llm(self.request) or random.choice(self.get_alive_agents())

    def finish(self) -> None:
        pass

    @timeout
    def action(self) -> str | None:
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case Request.DAY_START:
                self.day_start()
        return None
