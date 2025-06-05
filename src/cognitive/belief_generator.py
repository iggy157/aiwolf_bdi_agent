from typing import List
from jinja2 import Template
import json

from langchain_core.messages import HumanMessage
from cognitive.model_types import Belief

class BeliefGenerator:
    def __init__(self, agent):
        from agent.agent import Agent
        assert isinstance(agent, Agent)
        self.agent = agent

    def generate_beliefs(self, game_log: str) -> List[Belief]:
        """
        ゲームログから信念を生成する。
        game_log: ゲームの進行ログや状況をテキスト形式で渡す。
        """
        # config.yml の 'generate_belief' テンプレートを取得・描画する処理例（agent側実装依存）
        # ここでは自分でテンプレートを取得すると仮定して説明します。

        # 例として、agent.config['prompt']['generate_belief']がある想定
        prompt_template_str = self.agent.config["prompt"]["generate_belief"]
        template = Template(prompt_template_str)
        prompt = template.render(game_log=game_log)

        # メッセージ履歴にHumanMessageを追加し、LLM呼び出し
        self.agent.llm_message_history.append(HumanMessage(content=prompt))

        response = self.agent._send_message_to_llm("generate_belief")

        if response is None:
            return []

        try:
            belief_data = json.loads(response)
            beliefs: List[Belief] = []
            for item in belief_data:
                belief = Belief(
                    agent_id=item.get("agent_id"),
                    is_alive=item.get("is_alive", True),
                    divined=item.get("divined"),
                    mentioned_by=item.get("mentioned_by", 0),
                )
                beliefs.append(belief)
            return beliefs
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Belief parsing failed: {e}")
            return []
