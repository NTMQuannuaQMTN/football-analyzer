"""Machine-readable contracts (broadcast overlays, downstream consumers, LLM inputs)."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

EventType = Literal["PASS", "SHOT", "TACKLE", "PRESSURE", "POSSESSION_CHANGE", "GOAL",
                    "SUBSTITUTION", "RED_CARD", "KICK_OFF", "PLAYER_POSITION",
                    # labelled real-match events (SoccerNet)
                    "FOUL", "CORNER", "THROW_IN", "OFFSIDE", "FREE_KICK", "CLEARANCE", "BALL_OUT",
                    "CARD", "PENALTY", "SUBSTITUTION_LABEL", "TOUCH"]
MomentType = Literal["TACTICAL_TURNING_POINT", "PRESSURE_BREAK", "COUNTERATTACK", "FORMATION_SHIFT",
                     "RED_CARD_IMPACT", "SUBSTITUTION_IMPACT", "GOAL_BUILD_UP"]


class Event(BaseModel):
    model_config = ConfigDict(extra="allow")  # type-specific fields (target_x, xg, cause, ...)
    id: str
    timestamp: float
    event_type: EventType
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None


class Evidence(BaseModel):
    label: str
    detail: str
    value: Any
    unit: str
    timestamp: float
    source: str = "analytics"


class ChainStep(BaseModel):
    stage: Literal["EVENT", "STATE_CHANGE", "TACTICAL_CHANGE", "CONSEQUENCE", "OUTCOME"]
    label: str
    timestamp: float


class Replay(BaseModel):
    start: float
    end: float
    focus_players: List[str]
    focus_team: str
    overlay: str


class Moment(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    type: MomentType
    subtype: Optional[str] = None
    team: str
    timestamp: float
    end_timestamp: float
    minute_label: str
    title: str
    importance: float
    confidence: float
    trigger: Dict[str, Any]
    before: Dict[str, Any]
    after: Dict[str, Any]
    deltas: Dict[str, Any]
    evidence: List[Evidence]
    consequences: List[str]
    outcome: Dict[str, Any]
    chain: List[ChainStep]
    replay: Replay
    narrative: Dict[str, Any]
    counterfactual: Optional[Dict[str, Any]] = None
    links: List[Dict[str, str]]
