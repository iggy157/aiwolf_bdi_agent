"""エージェントを起動するためのモジュール."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from time import sleep
from utils.agent_utils import init_agent_from_packet

from aiwolf_nlp_common.client import Client
from aiwolf_nlp_common.packet import Request

# 認知エージェントの読み込み
from agent import Agent

if TYPE_CHECKING:
    from agent.agent import Agent

# ロガー設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def create_client(config: dict) -> Client:
    """クライアントの作成."""
    return Client(
        url=str(config["web_socket"]["url"]),
        token=(
            str(config["web_socket"]["token"])
            if config["web_socket"]["token"]
            else None
        ),
    )


def connect_to_server(client: Client, name: str) -> None:
    """サーバーへの接続処理."""
    while True:
        try:
            client.connect()
            logger.info("エージェント %s がゲームサーバに接続しました", name)
            break
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "エージェント %s がゲームサーバに接続できませんでした",
                name,
            )
            logger.warning(ex)
            logger.info("再接続を試みます")
            sleep(15)


def handle_game_session(
    client: Client,
    config: dict,
    name: str,
) -> None:
    """ゲームセッションの処理."""
    agent: Agent | None = None
    while True:
        packet = client.receive()
        if packet.request == Request.NAME:
            client.send(name)
            continue
        if packet.request == Request.INITIALIZE:
            # config.yml で Agent 利用を指定しているか判定
            # if config.get("agent", {}).get("use_cognitive_agent", False):
            #     role = config.get("agent", {}).get("role", "villager")
            #     agent = Agent(name, role=role)
            #     logger.info("Agent を使用（役職: %s）", role)
            # else:
            agent = init_agent_from_packet(config, name, packet)

        if not agent:
            raise ValueError(agent, "エージェントが初期化されていません")

        agent.set_packet(packet)
        req = agent.action()
        agent.agent_logger.packet(agent.request, req)
        if req:
            client.send(req)
        if packet.request == Request.FINISH:
            break


def connect(config: dict, idx: int = 1) -> None:
    """エージェントを起動する."""
    name = str(config["agent"]["team"]) + str(idx)
    while True:
        client = create_client(config)
        connect_to_server(client, name)
        try:
            handle_game_session(client, config, name)
        finally:
            client.close()
            logger.info("エージェント %s とゲームサーバの接続を切断しました", name)

        if not bool(config["web_socket"]["auto_reconnect"]):
            break
