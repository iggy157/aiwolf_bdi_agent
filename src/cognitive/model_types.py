from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Belief:
    agent_id: int
    is_alive: bool
    divined: Optional[str] = None  # 'HUMAN' or 'WEREWOLF'
    mentioned_by: int = 0

@dataclass
class Desire:
    type: str
    target_agent: Optional[int]
    score: float

@dataclass
class Intention:
    action_type: str
    target_agent: Optional[int]
    reason: str
