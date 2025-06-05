import os
import json
import yaml
from typing import List

from jinja2 import Template
from langchain_core.messages import HumanMessage
from cognitive.model_types import Intention, Belief, Desire

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

        prompt_payload = {
            "beliefs": belief_dicts,
            "desires": desire_dicts,
            "regulations": regulations
        }

        try:
            # config.ymlのgenerate_intentionテンプレートをJinja2で展開して使用
            prompt_template_str = self.agent.config["prompt"].get("generate_intention")
            if prompt_template_str:
                template = Template(prompt_template_str)
                prompt = template.render(
                    beliefs=json.dumps(belief_dicts, ensure_ascii=False),
                    desires=json.dumps(desire_dicts, ensure_ascii=False),
                    regulations=json.dumps(regulations, ensure_ascii=False),
                )
            else:
                prompt = json.dumps(prompt_payload, ensure_ascii=False)

            self.agent.llm_message_history.append(HumanMessage(content=prompt))
            response = self.agent._send_message_to_llm("generate_intention")
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Failed to send intention prompt: {e}")
            return []

        if response is None:
            return []

        try:
            parsed = json.loads(response)
            intentions: List[Intention] = [
                Intention(
                    action_type=item["action_type"],
                    target_agent=item.get("target_agent"),
                    reason=item.get("reason", "")
                )
                for item in parsed
            ]
            return intentions
        except Exception as e:
            self.agent.agent_logger.logger.error(f"Failed to parse LLM intention response: {e}")
            return []
