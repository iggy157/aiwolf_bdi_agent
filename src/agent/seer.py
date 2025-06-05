"""占い師のエージェントクラスを定義するモジュール."""

from __future__ import annotations

from aiwolf_nlp_common.packet import Role
from typing import Dict, List

from agent.agent import Agent


class Seer(Agent):
    """占い師のエージェントクラス."""

    def __init__(
        self,
        config: dict,
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """占い師のエージェントを初期化する."""
        super().__init__(config, name, game_id, role)
        
        # 占い師固有の属性
        self.divine_results: Dict[str, str] = {}  # {day: target}
        self.investigated_agents: List[str] = []

    def talk(self) -> str:
        """トークリクエストに対する応答を生成."""
        return super().talk()

    def divine(self) -> str:
        """占い対象を選択するメソッド."""
        # status_mapが空でないことを確認
        if not self.status_map:
            self.agent_logger.logger.warning("Status map is empty for Seer.")
            return ""

        # 生存エージェントのリストを取得
        alive_agents = [agent for agent, status in self.status_map.items() if status == Status.ALIVE]

        # 未調査の生存エージェントから選択
        target = next(
            (agent for agent in alive_agents if agent not in self.investigated_agents),
            None
        )

        # もし未調査のエージェントがいない場合、ランダムに選択
        if not target:
            target = self.select_random_alive_agent()

        # 選んだエージェントを調査済みリストに追加し、beliefsに保存
        self.investigated_agents.append(target)
        self.beliefs["last_divine_target"] = target

        return target

    def vote(self) -> str:
        """投票対象を決定."""
        # 信念と意図に基づく投票
        if "suspicious_agent" in self.beliefs:
            return self.beliefs["suspicious_agent"]
        return super().vote()

    def _create_reveal_message(self) -> str:
        """占い結果を公開するメッセージを生成."""
        if not self.divine_results:
            return "まだ占っていません"
            
        last_day = max(self.divine_results.keys())
        target = self.divine_results[last_day]
        return f"私は占い師です。{target}を占った結果、人狼でした。"
