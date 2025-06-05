"""村人のエージェントクラスを定義するモジュール."""

from __future__ import annotations

from aiwolf_nlp_common.packet import Role
from agent.agent import Agent


class Villager(Agent):
    """村人のエージェントクラス."""

    def __init__(
        self,
        config: dict,
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """村人のエージェントを初期化する."""
        super().__init__(config, name, game_id, role)
        # BDIモデル属性の初期化
        self.beliefs = []   # 信念（ゲーム状態の認識）
        self.desires = []   # 欲望（達成したい目標） 
        self.intentions = []  # 意図（実行する行動計画）

    def talk(self) -> str:
        """トークリクエストに対する応答を返す."""
        return super().talk()

    def vote(self) -> str:
        
        """投票リクエストに対する応答を生成"""
        # 投票ロジックの実装
        return super().vote()

    def _generate_speech(self) -> str:
        """信念/欲望/意図に基づく発話生成"""
        # 実際の自然言語生成ロジック（例：テンプレートベース）
        return "私は村人を信じています。"
