from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

"""
Smart Focus – ConcentrationEngine (v3)
====================================
Improvements over v2:
- STRESSED state from facial grimace detection (brow furrow, lip tension, jaw clench)
- SOCIAL_DISTRACTION as distinct state (not just generic DISTRACTED)
- Microsleep triggers instant DROWSY (bypasses hold timer)
- Hand-near-face contributes to FATIGUED signal
- Better alert messaging with grimace + hand context
- Weighted scoring uses grimace_score for stress component
"""


class FocusState(str, Enum):
    FOCUSED          = "FOCUSED"
    READING          = "READING"
    DISTRACTED       = "DISTRACTED"
    PHONE_USAGE      = "PHONE_USAGE"
    SELF_EXPLAINING  = "SELF_EXPLAINING"
    STRESSED         = "STRESSED"
    FATIGUED         = "FATIGUED"
    DROWSY           = "DROWSY"
    USER_ABSENT      = "USER_ABSENT"


@dataclass
class Alert:
    code:    str
    message: str
    active:  bool = False


@dataclass
class EngineOutput:
    state:          FocusState
    alerts:         list
    confidence:     float
    focus_score:    int
    raw_signals:    dict = field(default_factory=dict)


# ── Timing constants (seconds) ────────────────────────────────────────────────
STATE_HOLD       = 1.5
FATIGUE_HOLD     = 2.5
DISTRACT_HOLD    = 2.0
STRESS_HOLD      = 3.0   # grimace needs longer confirmation to avoid transient

# ── Score weights ─────────────────────────────────────────────────────────────
W_ATTENTION = 0.45
W_POSTURE   = 0.25
W_FATIGUE   = 0.30


class ConcentrationEngine:
    def __init__(self):
        self.state          = FocusState.USER_ABSENT
        self._state_since   = time.time()
        self._candidate     = None
        self._candidate_since = time.time()

        self._ema_score  = 75.0
        self._EMA_ALPHA  = 0.04   # slow gradual score change

        self._alerts = {
            "POSTURE":    Alert("ADJUST_POSTURE",      "Ajustez votre posture"),
            "PHONE":      Alert("PHONE_DETECTED",      "Posez votre téléphone"),
            "FATIGUE":    Alert("DROWSINESS_ALERT",    "Prenez une pause – signes de fatigue"),
            "MICROSLEEP": Alert("MICROSLEEP_ALERT",    "Micro-sommeil détecté !"),
            "YAWN":       Alert("YAWN_DETECTED",       "Bâillement détecté"),
            "SOCIAL":     Alert("SOCIAL_DISTRACTION",  "Distraction sociale détectée"),
            "DRIFT":      Alert("ATTENTION_DRIFT",     "Attention en dérive"),
            "EXPLAIN":    Alert("SELF_EXPLAINING",     "Mode auto-explication"),
            "READING":    Alert("READING_MODE",        "Mode lecture"),
            "STRESS":     Alert("STRESS_DETECTED",     "Signes de stress détectés"),
            "HAND_FACE":  Alert("HAND_NEAR_FACE",      "Main près du visage"),
            "HANDS_KNEE": Alert("HANDS_ON_KNEES",      "Mains sur les genoux"),
        }

    # ─── State transition with hysteresis ─────────────────────────────────────
    def _propose_state(self, candidate: FocusState, hold: float = STATE_HOLD) -> bool:
        now = time.time()
        if self._candidate != candidate:
            self._candidate       = candidate
            self._candidate_since = now
            return False
        return (now - self._candidate_since) >= hold

    def _set_state(self, new_state: FocusState):
        if new_state != self.state:
            self.state       = new_state
            self._state_since = time.time()
            self._candidate   = None

    def _set_alert(self, key: str, active: bool):
        if key in self._alerts:
            self._alerts[key].active = active

    # ─── Main update ──────────────────────────────────────────────────────────
    def update(
        self,
        attention:  dict,
        fatigue:    dict,
        posture:    dict,
        stress:     dict,
        phone:      dict,
    ) -> EngineOutput:
        # ── Extract signals ───────────────────────────────────────────────────
        face_present        = attention.get("face_present", False)
        is_distracted_att   = attention.get("is_distracted", False)
        is_reading          = attention.get("is_reading", False)
        social_present      = attention.get("social_present", False)
        social_distraction  = attention.get("social_distraction", False)
        is_speaking         = attention.get("is_speaking", False)
        num_faces           = attention.get("num_faces", 0)

        fatigue_score       = fatigue.get("fatigue_score", 0)
        fatigue_level       = fatigue.get("fatigue_level", "low")
        perclos             = fatigue.get("perclos", 0.0)
        microsleep          = fatigue.get("microsleep", False)
        yawn_in_progress    = fatigue.get("yawn_in_progress", False)
        yawn_count          = fatigue.get("yawn_count", 0)

        posture_state       = posture.get("posture_state", "good")
        bad_posture         = posture.get("bad_posture_confirmed", False)
        hands_on_knees      = posture.get("hands_on_knees", False)
        hand_near_face      = posture.get("hand_near_face", False)

        agitation           = stress.get("agitation_score", 0)
        grimace_score       = stress.get("grimace_score", 0)

        phone_detected      = phone.get("phone_detected", False)
        phone_distracting   = phone.get("phone_distracting", False)

        # ── Determine candidate state (priority order) ────────────────────────

        # 1. No face
        if not face_present:
            candidate = FocusState.USER_ABSENT
            hold      = 0.5

        # 2. Microsleep → instant DROWSY (no hold)
        elif microsleep:
            candidate = FocusState.DROWSY
            hold      = 0.3

        # 3. Severe drowsiness / high PERCLOS
        elif perclos > 40.0 or fatigue_score > 75:
            candidate = FocusState.DROWSY
            hold      = FATIGUE_HOLD

        # 4. Moderate fatigue (needs multiple yawns or high score)
        elif fatigue_score > 50 or (fatigue_level == "moderate" and yawn_count >= 2):
            candidate = FocusState.FATIGUED
            hold      = FATIGUE_HOLD

        # 5. Phone distraction
        elif phone_distracting:
            candidate = FocusState.PHONE_USAGE
            hold      = DISTRACT_HOLD

        # Stress candidate removed per user request
        elif False: # was Stressed
            candidate = FocusState.STRESSED
            hold      = STRESS_HOLD

        # 7. Distraction (gaze away OR social distraction — merged)
        elif social_distraction or (is_distracted_att and not is_reading):
            candidate = FocusState.DISTRACTED
            hold      = DISTRACT_HOLD

        # 10. Self-explaining
        elif is_speaking and num_faces == 1:
            candidate = FocusState.SELF_EXPLAINING
            hold      = 1.2

        # 11. Reading
        elif is_reading:
            candidate = FocusState.READING
            hold      = 1.0

        # 12. Default: focused
        else:
            candidate = FocusState.FOCUSED
            hold      = STATE_HOLD

        if self._propose_state(candidate, hold):
            self._set_state(candidate)

        # ── Alert flags ──────────────────────────────────────────────────────
        self._set_alert("POSTURE",    bad_posture)
        self._set_alert("PHONE",      phone_detected)
        self._set_alert("FATIGUE",    fatigue_score > 55 or perclos > 30)
        self._set_alert("MICROSLEEP", microsleep)
        self._set_alert("YAWN",       yawn_in_progress)
        self._set_alert("SOCIAL",     social_distraction)
        self._set_alert("DRIFT",      is_distracted_att and not social_present and not phone_distracting)
        self._set_alert("EXPLAIN",    self.state == FocusState.SELF_EXPLAINING)
        self._set_alert("READING",    self.state == FocusState.READING)
        self._set_alert("STRESS",     grimace_score > 45)
        self._set_alert("HAND_FACE",  hand_near_face)
        self._set_alert("HANDS_KNEE", hands_on_knees)

        # ── Focus score ──────────────────────────────────────────────────────
        att_ok = 0 if is_distracted_att else 100
        if is_reading:
            att_ok = 90
        if self.state == FocusState.SELF_EXPLAINING:
            att_ok = 85

        pos_ok = posture.get("posture_score", 70)
        fat_ok = max(0, 100 - fatigue_score)

        instant_score = (
            W_ATTENTION * att_ok
            + W_POSTURE   * pos_ok
            + W_FATIGUE   * fat_ok
        )

        # Penalties are mild — score drops gradually via EMA
        if phone_distracting:
            instant_score *= 0.80
        if social_distraction:
            instant_score *= 0.85
        if microsleep:
            instant_score *= 0.50
        # Stress penalty removed
        if phone_distracting:
            instant_score *= 0.80
        if social_distraction:
            instant_score *= 0.85
        if microsleep:
            instant_score *= 0.50

        self._ema_score = self._EMA_ALPHA * instant_score + (1 - self._EMA_ALPHA) * self._ema_score

        active_alerts = [a for a in self._alerts.values() if a.active]

        return EngineOutput(
            state       = self.state,
            alerts      = active_alerts,
            confidence  = 1.0,
            focus_score = int(self._ema_score),
            raw_signals = {
                "attention":   attention,
                "fatigue":     fatigue,
                "posture":     posture,
                "stress":      stress,
                "phone":       phone,
            },
        )
