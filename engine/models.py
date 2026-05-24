from pydantic import BaseModel, Field
from typing import List, Optional

"""
Smart Focus - Output Models (JSON structure)
Represents the exact 4-level data output format requested.
"""

class PresenceInfo(BaseModel):
    main_person_present: bool = False
    person_count: int = 0
    face_detected: bool = False

class InstantObservations(BaseModel):
    head_direction: str = "frontal" # e.g., 'frontal', 'left', 'right', 'up', 'down'
    gaze_zone: str = "screen"       # e.g., 'screen', 'desk', 'away'
    eyes_state: str = "open"        # e.g., 'open', 'closed', 'squinting'
    posture_state: str = "unknown"  # 'good', 'warning', 'bad' (instantaneous)
    phone_detected: bool = False
    face_tension_level: str = "low" # 'low', 'moderate', 'high' based on raw grimace
    agitation_level: str = "low"    # 'low', 'moderate', 'high' based on raw jitter

class ShortWindowInference(BaseModel):
    possible_reading: bool = False
    possible_writing: bool = False
    possible_thinking: bool = False
    possible_self_explaining: bool = False
    possible_social_interaction: bool = False
    possible_phone_distraction: bool = False
    possible_fatigue: bool = False

class ConsolidatedStates(BaseModel):
    work_mode: str = "focused" # 'focused', 'focused_reading', 'focused_writing', 'thinking', 'self_explaining', 'brief_off_task', 'phone_distraction', 'social_distraction'
    attention_state: str = "focused" # 'focused', 'slightly_distracted', 'distracted'
    fatigue_state: str = "normal"  # 'normal', 'fatigue_warning', 'fatigue_high'
    social_state: str = "alone"    # 'alone', 'other_person_present', 'active_interaction'
    phone_state: str = "not_detected" # 'not_detected', 'detected_not_used', 'probable_in_use'
    posture_state: str = "good"    # 'good', 'acceptable', 'poor_persistent'
    reasoning_indices: List[str] = Field(default_factory=list) # e.g. ["low_agitation", "looking_down", "no_social"]

class Reliability(BaseModel):
    work_mode_confidence: float = 0.0
    attention_confidence: float = 0.0
    fatigue_confidence: float = 0.0

class TemporalContext(BaseModel):
    observed_for_sec: float = 0.0
    stable_state_for_sec: float = 0.0

class AlertStatus(BaseModel):
    should_alert: bool = False
    alert_type: str = "none"   # e.g., 'fatigue', 'phone'
    severity: str = "none"     # 'none', 'low', 'medium', 'high'
    reason: Optional[str] = None
    confidence: float = 0.0
    duration: float = 0.0

class CVOutputPayload(BaseModel):
    session_id: str
    timestamp: str
    presence: PresenceInfo
    instant_observations: InstantObservations
    short_window_inference: ShortWindowInference
    consolidated_states: ConsolidatedStates
    reliability: Reliability
    temporal_context: TemporalContext
    alert: AlertStatus

    # Extra numeric metrics used for clean JSON + session analytics.
    # Kept out of UI by default.
    metrics: dict = Field(default_factory=dict)

    events: List[dict] = Field(default_factory=list)
