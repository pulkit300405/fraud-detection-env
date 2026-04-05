"""
models.py - typed models for fraud detection env
action + observation inheriting from openenv base types
"""
from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class FraudDetectAction(Action):
    action: str = Field(..., description="investigate:<signal> or verdict:<fraud|real>")
    reasoning: Optional[str] = Field(default=None, description="Why agent made this choice")

class FraudDetectObservation(Observation):
    session_id: str = Field(default="", description="Session being investigated")
    logs: List[Dict[str, Any]] = Field(default_factory=list, description="App behavior log entries")
    step_num: int = Field(default=0, description="Current episode step")
    signals_revealed: Dict[str, Any] = Field(default_factory=dict, description="Signals investigated so far")
    available_actions: List[str] = Field(default_factory=list, description="Valid actions this step")
    task: str = Field(default="", description="flag-obvious | explain-subtle | adversarial-hunt")
    difficulty: str = Field(default="easy", description="easy | medium | hard")
    context_hint: Optional[str] = Field(default=None, description="Hint shown in easy tier only")
    ground_truth: Optional[str] = Field(default=None, description="Revealed after episode ends")