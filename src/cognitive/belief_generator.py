from typing import List
import json
from cognitive.model_types import Belief

class BeliefGenerator:
    def __init__(self, agent):
        from agent.agent import Agent
        assert isinstance(agent, Agent)
        self.agent = agent

    def generate_beliefs(self, bdi_data: List[dict]) -> List[Belief]:
        """
        BDI分析結果から信念を生成する（LLMによる要約・再構成あり）。
        bdi_data: bdi_analysis.pyで生成されたBDI分析データ
        """
        # config.ymlの'generate_belief'テンプレートにbdi_dataを渡す
        prompt_template_str = self.agent.config["prompt"]["generate_belief"]
        from jinja2 import Template
        template = Template(prompt_template_str)
        prompt = template.render(bdi_data=bdi_data)

        # LLMを使って信念を整形・生成
        response = self.agent._send_message_to_llm("generate_belief", prompt)

        if response is None:
            return []

        try:
            belief_data = json.loads(response).get("beliefs", [])
            beliefs: List[Belief] = []
            for item in belief_data:
                belief = Belief(
                    agent_id=item.get("agent_id"),
                    is_alive=item.get("is_alive", True),
                    divined=item.get("divined"),
                    mentioned_by=item.get("mentioned_by", 0),
                    type=item.get("type", "unknown"),
                    content=item.get("content", "")
                )
                beliefs.append(belief)
            return beliefs
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Belief parsing failed: {e}")
            return []
