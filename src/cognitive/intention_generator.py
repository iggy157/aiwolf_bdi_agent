import os
import json
import yaml
from typing import List

from jinja2 import Template
from langchain_core.messages import HumanMessage
from cognitive.model_types import Intention, Belief, Desire
from transformers import pipeline

class IntentionGenerator:
    def __init__(self, agent):
        from agent.agent import Agent
        from cognitive.belief_generator import BeliefGenerator
        from cognitive.desire_generator import DesireGenerator
        assert isinstance(agent, Agent)
        self.agent = agent
        self.belief_generator = BeliefGenerator(agent)
        self.desire_generator = DesireGenerator(agent)
        self.config_dir = os.path.join(os.path.dirname(__file__), "../config/regulation")

        # NLIモデルの読み込み
        self.nlp = pipeline("zero-shot-classification")

    def _load_regulations(self, role: str) -> List[dict]:
        file_path = os.path.join(self.config_dir, f"{role}.yml")
        if not os.path.exists(file_path):
            self.agent.agent_logger.logger.warning(f"Regulation file not found: {file_path}")
            return []
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("regulations", [])

    def generate_intentions(self, beliefs: List[Belief] = None, desires: List[Desire] = None) -> List[Intention]:
        # beliefs, desiresは引数で受け取る形にして柔軟性を持たせる
        if beliefs is None:
            beliefs = self.belief_generator.generate_beliefs()
        if desires is None:
            desires = self.desire_generator.generate_desires()

        regulations = self._load_regulations(self.agent.role.lower())

        belief_dicts = [vars(b) for b in beliefs]
        desire_dicts = [vars(d) for d in desires]

        # プランルールを適切に選択し、実行可能な意図を生成する
        intentions = []

        for regulation in regulations:
            # ルールのIF部分に基づいて、適切なプランコンテキストを設定
            if self._evaluate_condition_with_nli(regulation, belief_dicts):
                intention = self._create_intention_from_regulation(regulation, belief_dicts, desire_dicts)
                intentions.append(intention)

        return intentions

    def _evaluate_condition_with_nli(self, regulation: dict, belief_dicts: List[dict]) -> bool:
        """NLIを使用してプランルールの条件部分（IF）を評価する"""
        if "IF" in regulation:
            # NLIを使用して、belief_dictsが規範の条件（プランコンテキスト）を満たすか評価
            condition_statement = regulation["IF"]
            nli_result = self._perform_nli(condition_statement, belief_dicts)
            return nli_result  # NLIがTrueの場合のみプランが適用される
        return False

    def _perform_nli(self, condition_statement: str, belief_dicts: List[dict]) -> bool:
        """NLIモデルを使用して条件文が信念ベースに含意されるかを判断"""
        belief_text = " ".join([belief["content"] for belief in belief_dicts])  # belief_dictsから文章を生成

        # NLI（zero-shot-classification）で信念と条件文の一致を評価
        result = self.nlp(condition_statement, candidate_labels=[belief_text])

        # 最も一致するスコアが高いものが条件に合致しているか確認
        return result["scores"][0] > 0.5  # 閾値0.5を使って合致度を判定

    def _create_intention_from_regulation(self, regulation: dict, belief_dicts: List[dict], desire_dicts: List[dict]) -> Intention:
        """プランルールから意図を生成する"""
        action_type = regulation.get("action_type", "action")
        target_agent = regulation.get("target_agent", None)
        reason = regulation.get("reason", "No specific reason")

        return Intention(
            action_type=action_type,
            target_agent=target_agent,
            reason=reason
        )
