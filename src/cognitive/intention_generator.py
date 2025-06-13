import os
import yaml
import random
import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from cognitive.model_types import Intention, Belief, Desire

class RLPlanEnvironment(Env):
    """
    強化学習環境。状態はbelief/desireテキストの埋め込みベクトル、行動はregulationのTHEN文のID。
    報酬は状態（信念・欲求）がregulationのIF文と近く、かつ選択行動がTHEN文と近いほど高くなる。
    """
    def __init__(self, regulations: List[dict]):
        super().__init__()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.regulations = [reg for reg in regulations if 'IF' in reg and 'THEN' in reg]
        self.action_patterns = [reg['THEN'] for reg in self.regulations]
        self.action_space = Discrete(len(self.action_patterns))
        self.if_embeddings = [self.model.encode(reg['IF']) for reg in self.regulations]
        self.then_embeddings = [self.model.encode(reg['THEN']) for reg in self.regulations]
        # 観測空間: 埋め込み次元
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.model.get_sentence_embedding_dimension(),),
            dtype=np.float32
        )
        self.state_embedding = None

    def reset(self, belief_texts=None, desire_texts=None):
        # 状態をbelief/desireのcontentテキストを連結したベクトルに
        belief_text = " ".join(belief_texts) if belief_texts else ""
        desire_text = " ".join(desire_texts) if desire_texts else ""
        combined = f"{belief_text} {desire_text}"
        self.state_embedding = self.model.encode(combined)
        return self.state_embedding

    def step(self, action_idx):
        # 選択したTHEN文（行動パターン）
        chosen_action_emb = self.then_embeddings[action_idx]
        # 最大reward計算
        max_reward = 0.0
        for i, reg in enumerate(self.regulations):
            if_sim = self._cosine_similarity(self.state_embedding, self.if_embeddings[i])
            then_sim = self._cosine_similarity(chosen_action_emb, self.then_embeddings[i])
            reward = (if_sim + then_sim) / 2
            max_reward = max(max_reward, reward)
        done = True  # 1ターンで終了
        info = {"selected_action": self.action_patterns[action_idx]}
        return self.state_embedding, max_reward, done, info

    def _cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))


class IntentionGenerator:
    def __init__(self, agent, train_steps: int = 5):
        from agent.agent import Agent
        from cognitive.belief_generator import BeliefGenerator
        from cognitive.desire_generator import DesireGenerator
        assert isinstance(agent, Agent)
        self.agent = agent
        self.belief_generator = BeliefGenerator(agent)
        self.desire_generator = DesireGenerator(agent)
        self.config_dir = os.path.join(os.path.dirname(__file__), "../config/regulation")

        # regulationファイル読込＆環境セットアップ
        regulations = self._load_regulations(self.agent.role.lower())
        self.env = RLPlanEnvironment(regulations)
        self.regulations = regulations

        # PPO 強化学習モデル準備
        self.model = PPO("MlpPolicy", self.env, verbose=1)

        # 強化学習モデルの訓練
        print("==== 強化学習モデルの訓練開始 ====")
        # 訓練用サンプル状態を生成する仮関数
        train_states = self._sample_training_states(20)
        for state in train_states:
            self.env.reset(*state)
        self.model.learn(total_timesteps=train_steps)
        print("==== 強化学習モデルの訓練終了 ====")

    def _load_regulations(self, role: str) -> List[dict]:
        file_path = os.path.join(self.config_dir, f"{role}.yml")
        if not os.path.exists(file_path):
            self.agent.agent_logger.logger.warning(f"Regulation file not found: {file_path}")
            return []
        with open(file_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("regulations", [])

    def _sample_training_states(self, num_samples: int = 10):
        """
        訓練用に規範ファイルからbelief/desireの例を生成（例なのでランダムでOK）
        """
        belief_examples = []
        desire_examples = []
        for reg in self.regulations:
            if 'IF' in reg and 'THEN' in reg:
                belief_examples.append([reg['IF']])
                desire_examples.append([reg['THEN']])
        # ダミーデータ生成
        samples = []
        for _ in range(num_samples):
            b = random.choice(belief_examples) if belief_examples else [""]
            d = random.choice(desire_examples) if desire_examples else [""]
            samples.append((b, d))
        return samples

    def generate_intentions(self, beliefs: List[Belief] = None, desires: List[Desire] = None) -> List[Intention]:
        if beliefs is None:
            beliefs = self.belief_generator.generate_beliefs()
        if desires is None:
            desires = self.desire_generator.generate_desires()
        belief_dicts = [vars(b) for b in beliefs]
        desire_dicts = [vars(d) for d in desires]
        intentions = []

        # まずNLI比較・ルールベース意図生成を試す
        for regulation in self.regulations:
            if self._evaluate_condition_with_nli(regulation, belief_dicts, desire_dicts):
                intention = self._create_intention_from_regulation(regulation, belief_dicts, desire_dicts)
                intentions.append(intention)

        # NLIで適用できるルールがなければRL fallback
        if not intentions:
            # RLで意図生成（ここは既存RL実装の呼び出し）
            belief_texts = [b["content"] for b in belief_dicts]
            desire_texts = [d["content"] for d in desire_dicts]
            obs = self.env.reset(belief_texts=belief_texts, desire_texts=desire_texts)
            action_idx, _ = self.model.predict(obs)
            action_idx = int(action_idx)
            selected_then = self.env.action_patterns[action_idx]
            action_type = "generated_intention"
            reason = f"Generated by RL (selected: {selected_then})"
            intention = Intention(
                action_type=action_type,
                target_agent=None,
                reason=reason
            )
            intentions.append(intention)

        return intentions

    def _evaluate_condition_with_nli(self, regulation: dict, belief_dicts: List[dict], desire_dicts: List[dict]) -> bool:
        """NLIを使用してプランルールの条件部分（IF）を評価する（belief+desireで趣旨比較）"""
        if "IF" in regulation:
            condition_statement = regulation["IF"]
            nli_result = self._perform_nli(condition_statement, belief_dicts, desire_dicts)
            return nli_result  # NLIがTrueの場合のみプランが適用される
        return False

    def _perform_nli(self, condition_statement: str, belief_dicts: List[dict], desire_dicts: List[dict]) -> bool:
        """
        NLIモデルを使用して条件文が信念・欲求ベースに含意されるかを判断。
        beliefとdesireを組み合わせて比較対象にする。
        """
        belief_text = " ".join([belief["content"] for belief in belief_dicts])
        desire_text = " ".join([desire["content"] for desire in desire_dicts])
        combined_text = f"{belief_text} {desire_text}"

        # NLI（zero-shot-classification）で信念+欲求と条件文の一致を評価
        result = self.nlp(condition_statement, candidate_labels=[combined_text])

        # 最も一致するスコアが条件に合致しているか確認
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

    def _generate_intention_from_rl_model(self, belief_dicts: List[dict], desire_dicts: List[dict]) -> Intention:
        """強化学習モデルから意図を生成する"""
        state = self.env.reset()  # 初期状態を取得
        action = self.model.predict(state)[0]  # 強化学習モデルで最適な行動を予測

        # 行動を基に意図を生成
        action_type = f"action_{action}"
        target_agent = None  # ここでターゲットエージェントを決定するロジックを追加
        reason = "Generated by RL model"

        return Intention(
            action_type=action_type,
            target_agent=target_agent,
            reason=reason
        )
