import os
import json
import yaml
from typing import List

from jinja2 import Template
from langchain_core.messages import HumanMessage
from cognitive.model_types import Desire, Belief

class DesireGenerator:
    def __init__(self, agent):
        from agent.agent import Agent
        from cognitive.belief_generator import BeliefGenerator
        assert isinstance(agent, Agent)
        self.agent = agent
        self.belief_generator = BeliefGenerator(agent)
        self.config_dir = os.path.join(os.path.dirname(__file__), "../config/instinct")

    def _load_instincts(self, role: str) -> List[dict]:
        file_path = os.path.join(self.config_dir, f"{role}.yml")
        if not os.path.exists(file_path):
            self.agent.agent_logger.logger.warning(f"Instinct file not found: {file_path}")
            return []
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("instincts", [])

    def generate_desires(self, beliefs: List[Belief] = None) -> List[Desire]:
        # beliefsを引数で受け取る形にして柔軟化（未指定なら生成）
        if beliefs is None:
            beliefs = self.belief_generator.generate_beliefs()

        instincts = self._load_instincts(self.agent.role.lower())

        belief_dicts = [vars(b) for b in beliefs]
        prompt_payload = {
            "beliefs": belief_dicts,
            "instincts": instincts
        }

        try:
            # config.yml の 'generate_desire' テンプレートを展開して渡す例（agent.configがある場合）
            prompt_template_str = self.agent.config["prompt"].get("generate_desire")
            if prompt_template_str:
                template = Template(prompt_template_str)
                prompt = template.render(
                    beliefs=json.dumps(belief_dicts, ensure_ascii=False),
                    instincts=json.dumps(instincts, ensure_ascii=False)
                )
            else:
                # もしテンプレートがなければJSONを直接渡すだけにする
                prompt = json.dumps(prompt_payload, ensure_ascii=False)

            self.agent.llm_message_history.append(HumanMessage(content=prompt))
            response = self.agent._send_message_to_llm("generate_desire")
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Failed to generate desire prompt: {e}")
            return []

        if response is None:
            return []

        try:
            parsed = json.loads(response)
            desires = [
                Desire(
                    type=item["type"],
                    target=item.get("target"),
                    priority=item.get("priority", 0.0),
                    reason=item.get("reason", "")
                )
                for item in parsed
            ]
            return desires
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Failed to parse LLM desire response: {e}")
            return []
