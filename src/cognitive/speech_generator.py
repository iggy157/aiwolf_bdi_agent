import json
from typing import List

from jinja2 import Template
from langchain_core.messages import HumanMessage
from cognitive.model_types import Belief, Desire, Intention

class SpeechGenerator:
    def __init__(self, agent):
        from agent.agent import Agent
        from cognitive.belief_generator import BeliefGenerator
        from cognitive.desire_generator import DesireGenerator
        from cognitive.intention_generator import IntentionGenerator
        assert isinstance(agent, Agent)
        self.agent = agent
        self.belief_generator = BeliefGenerator(agent)
        self.desire_generator = DesireGenerator(agent)
        self.intention_generator = IntentionGenerator(agent)

    def generate_speech(self,
                        beliefs: List[Belief] = None,
                        desires: List[Desire] = None,
                        intentions: List[Intention] = None
                       ) -> str:
        # 事前に生成済みのBDIデータを使えるよう引数受け取り可能に
        if beliefs is None:
            beliefs = self.belief_generator.generate_beliefs()
        if desires is None:
            desires = self.desire_generator.generate_desires(beliefs)
        if intentions is None:
            intentions = self.intention_generator.generate_intentions(beliefs, desires)

        payload = {
            "beliefs": [vars(b) for b in beliefs],
            "desires": [vars(d) for d in desires],
            "intentions": [vars(i) for i in intentions]
        }

        try:
            # config.yml の generate_speech テンプレート取得
            prompt_template_str = self.agent.config["prompt"].get("generate_speech")
            if prompt_template_str:
                template = Template(prompt_template_str)
                prompt = template.render(
                    beliefs=json.dumps(payload["beliefs"], ensure_ascii=False),
                    desires=json.dumps(payload["desires"], ensure_ascii=False),
                    intentions=json.dumps(payload["intentions"], ensure_ascii=False),
                )
            else:
                prompt = json.dumps(payload, ensure_ascii=False)

            self.agent.llm_message_history.append(HumanMessage(content=prompt))
            response = self.agent._send_message_to_llm("generate_speech")
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Failed to generate speech prompt: {e}")
            return ""

        if response is None:
            return ""

        return response.strip()
