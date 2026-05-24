"""
TemporalEngine - Refactored Orchestrator.
Coordinates the modular pipeline: L1 (Analyzers) -> L2 (Scores) -> L3 (States) -> Fusion -> L4 (Alerts).
"""

from __future__ import annotations
import time
import numpy as np
from config.cv_config import config

# Modular Components
from engine.score_manager import ScoreManager
from engine.state_manager import StateManager
from engine.fusion_engine import FusionEngine
from engine.alert_manager import AlertManager

from engine.models import (
    AlertStatus, CVOutputPayload, ConsolidatedStates, 
    InstantObservations, PresenceInfo, Reliability, 
    ShortWindowInference, TemporalContext
)

class TemporalEngine:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._start_time = time.time()
        
        # Initialize Layer Managers
        self.score_manager = ScoreManager(fps=float(config.TARGET_FPS), smoothing_sec=config.SMOOTHING_WINDOW_SECONDS)
        self.state_manager = StateManager()
        self.alert_manager = AlertManager()
        
        # Persistence for work mode and distraction
        self._state_start = time.time()
        self._distraction_acc: float = 0.0
        self._last_time = time.time()

    def _head_direction(self, yaw: float, pitch: float) -> str:
        # Use a more tolerant center zone to avoid micro-movements triggering distraction
        if yaw > config.YAW_DISTRACT_THRESH_DEG: return "right"
        if yaw < -config.YAW_DISTRACT_THRESH_DEG: return "left"
        if pitch > config.PITCH_UP_THRESH_DEG: return "up"
        if pitch < -config.PITCH_DOWN_THRESH_DEG: return "down"
        
        # Buffer zone: if within tolerance, consider it frontal
        if abs(yaw) < config.YAW_TOLERANCE_CENTER and abs(pitch) < config.YAW_TOLERANCE_CENTER:
            return "frontal"
            
        return "frontal_relaxed"

    def process(self, att: dict, fat: dict, pos: dict, phone: dict) -> CVOutputPayload:
        now = time.time()
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        # 1. RAW DATA (L1) - Already parsed by main_cv and passed here
        face_present = bool(att.get("face_present", False))
        num_faces = int(att.get("num_faces", 0) or 0)
        yaw, pitch = float(att.get("yaw", 0.0)), float(att.get("pitch", 0.0))
        mar = float(att.get("mar", 0.0))
        phone_found = bool(phone.get("phone_found", False))
        phone_conf = float(phone.get("confidence", 0.0))
        hand_face = bool(pos.get("hand_near_face", False))
        posture_raw = float(pos.get("posture_score", 100.0) or 100.0)
        bad_posture_flag = bool(pos.get("bad_posture_confirmed", False))
        
        head_dir = self._head_direction(yaw, pitch)
        
        if not face_present:
            # Handle absence gracefully
            return self._create_empty_payload(now, timestamp, face_present, num_faces, head_dir)

        # 2. SCORE COMPUTATION (L2)
        # Prepare evidence for ScoreManager
        is_tilting = abs(yaw) > 18.0 or abs(pitch) > 18.0 or float(pos.get("tilt_score", 1.0)) < 0.85
        
        # Extraction logic
        # A. Fatigue Multi-factor calculation (L2)
        eye_fatigue = float(fat.get("fatigue_score", 0.0)) # Already includes perclos/blinks
        yawn_freq   = float(fat.get("yawn_frequency_per_min", 0.0))
        yawn_sig    = min(1.0, yawn_freq / 2.0) * 100.0    # 2 yawns/min = 100% signal
        
        posture_sig = max(0.0, 100.0 - posture_raw)        # Low posture score = high fatigue signal
        
        # Behavioral signals
        hands_knee = bool(pos.get("hands_on_knees", False))
        behavior_sig = 0.0
        if hands_knee: behavior_sig += 60.0
        if not face_present: behavior_sig += 20.0 # Just in case
        
        # Weighted Fusion
        fatigue_unified = (
            (eye_fatigue  * config.WEIGHT_FATIGUE_EYES) +
            (yawn_sig     * config.WEIGHT_FATIGUE_YAWN) +
            (posture_sig  * config.WEIGHT_FATIGUE_POSTURE) +
            (behavior_sig * config.WEIGHT_FATIGUE_BEHAVIOR)
        )
        
        raw_evidence = {
            "reading_ev": config.WEIGHT_READING_PITCH_DOWN if head_dir == "down" else 0.0,
            "writing_ev": config.WEIGHT_WRITING_PITCH_DOWN if head_dir == "down" else 0.0,
            "thinking_ev": config.WEIGHT_THINKING_YAW_AWAY if head_dir in ("up", "left", "right") else 0.0,
            "speech_ev": 0.4 if mar > config.LIP_MAR_SPEECH_THRESH else 0.0,
            "social_ev": 1.0 if num_faces >= 2 else 0.0,
            "phone_ev": 0.8 if phone_found else 0.0,
            "distracted_ev": 0.0, # Computed below
            "fatigue_sig": fatigue_unified,
            "posture_raw": posture_raw
        }
        
        # Phone Heuristic Fusion
        if hand_face and is_tilting:
            raw_evidence["phone_ev"] = max(raw_evidence["phone_ev"], 0.75 if phone_found else 0.45)
            
        # Time delta
        dt = min(now - self._last_time, 0.5)
        self._last_time = now

        # Gaze Distraction Logic (Refined: Progressive Spatial & Temporal Accumulator)
        # Distraction is triggered if looking AWAY from the workspace (Left, Right, Up)
        # Looking DOWN is considered "Desk/Notes" and is part of the focus zone.
        is_looking_away = head_dir in ("left", "right", "up")
        
        # We don't accumulate basic distraction if it's a social or phone event
        if is_looking_away and not phone_found and num_faces < 2:
            # Build distraction time (Max bounded)
            self._distraction_acc = min(self._distraction_acc + dt, config.DISTRACTION_WINDOW_SEC * 1.5)
        else:
            # Decay the distraction memory (Decays 1.5x faster than it builds)
            # This prevents a quick glance at the screen from resetting a 5-second stare
            self._distraction_acc = max(self._distraction_acc - dt * 1.5, 0.0)
            
        # Progressive evidence based on the accumulated duration
        if self._distraction_acc >= config.DISTRACTION_WINDOW_SEC: 
            raw_evidence["distracted_ev"] = 0.80 # Confirmed Deep Distraction
        elif self._distraction_acc >= config.DISTRACTION_WINDOW_SEC * 0.4:
            raw_evidence["distracted_ev"] = 0.45 # Slightly Distracted
        else:
            raw_evidence["distracted_ev"] = 0.0
            
        # Task Evidence: Looking DOWN contributes to Reading/Writing focus
        if head_dir == "down":
            raw_evidence["reading_ev"] = config.WEIGHT_READING_PITCH_DOWN
            raw_evidence["writing_ev"] = config.WEIGHT_WRITING_PITCH_DOWN

        smoothed_scores = self.score_manager.compute_scores(raw_evidence)

        # 3. SUB-STATE COMPUTATION (L3)
        sub_states = self.state_manager.compute_sub_states(now, smoothed_scores, {"bad_posture_confirmed": bad_posture_flag, "posture_raw": posture_raw})

        # 4. GLOBAL STATE FUSION (Using stable sub-states from hysteresis)
        active_tags = []
        if sub_states["fatigue"] == "drowsy":
            active_tags.append("drowsy")
        elif sub_states["fatigue"] == "fatigued":
            active_tags.append("fatigued")
        elif sub_states["fatigue"] == "slightly_fatigued":
            active_tags.append("slightly_fatigued")
            
        if sub_states["posture"] == "poor_persistent":
            active_tags.append("poor_posture")
        if sub_states["phone"] == "probable_in_use":
            active_tags.append("phone_detected")
        if sub_states["distraction"] == "distracted":
            active_tags.append("distracted")
        elif sub_states["distraction"] == "slightly_distracted":
            active_tags.append("slightly_distracted")
        
        # Workspace Task Tags
        if sub_states.get("task") == "reading":
            active_tags.append("reading")
        elif sub_states.get("task") == "writing":
            active_tags.append("writing")
            
        if num_faces >= 2:
            active_tags.append("social_interaction")
        
        global_mode = FusionEngine.compute_global_state(active_tags, smoothed_scores)
        
        # 5. ALERT EVALUATION (L4)
        cons_states = ConsolidatedStates(
            work_mode=global_mode,
            fatigue_state=sub_states["fatigue"],
            posture_state=sub_states["posture"],
            # ... other fields filled below
        )
        alert_status = self.alert_manager.evaluate(cons_states)
        
        # 6. PAYLOAD GENERATION
        return self._build_payload(now, timestamp, global_mode, sub_states, smoothed_scores, alert_status, active_tags, head_dir, phone_found, phone_conf, hand_face, num_faces, is_tilting)

    def _create_empty_payload(self, now, timestamp, face_present, num_faces, head_dir):
        # Stable fallback for missing face
        return CVOutputPayload(
            session_id=self.session_id, timestamp=timestamp,
            presence=PresenceInfo(main_person_present=face_present, person_count=num_faces, face_detected=face_present),
            instant_observations=InstantObservations(head_direction=head_dir, gaze_zone="none", eyes_state="unknown", posture_state="unknown", phone_detected=False, face_tension_level="low", agitation_level="low"),
            consolidated_states=ConsolidatedStates(work_mode="missing_face", fatigue_state="normal", posture_state="acceptable", reasoning_indices=["face_missing"]),
            metrics={"fatigue_score": 0.0, "posture_score": 70.0, "distraction_risk": 0.0},
            temporal_context=TemporalContext(observed_for_sec=round(now - self._start_time, 2), stable_state_for_sec=0.0),
            alert=AlertStatus(),
            short_window_inference=ShortWindowInference(),
            reliability=Reliability()
        )

    def _build_payload(self, now, timestamp, global_mode, sub_states, scores, alert, tags, head_dir, phone_found, phone_conf, hand_face, num_faces, is_tilting):
        reasons = tags # Simple mapping for reasoning indices
        
        cons_states = ConsolidatedStates(
            work_mode=global_mode,
            attention_state="distracted" if global_mode in ("phone_distraction", "social_distraction") else "focused",
            fatigue_state=sub_states["fatigue"],
            social_state="active_interaction" if scores["social"] > 0.7 else "alone",
            phone_state=sub_states["phone"],
            posture_state=sub_states["posture"],
            reasoning_indices=reasons
        )
        
        return CVOutputPayload(
            session_id=self.session_id, timestamp=timestamp,
            presence=PresenceInfo(main_person_present=True, person_count=num_faces, face_detected=True),
            instant_observations=InstantObservations(head_direction=head_dir, gaze_zone="screen", eyes_state="open", posture_state=sub_states["posture"], phone_detected=phone_found, face_tension_level="low", agitation_level="low"),
            consolidated_states=cons_states,
            metrics={"fatigue_score": round(scores["ema_fatigue"], 1), "posture_score": round(scores["ema_posture"], 1), "distraction_risk": round(scores["distracted"], 2)},
            temporal_context=TemporalContext(observed_for_sec=round(now - self._start_time, 2), stable_state_for_sec=round(now - self._state_start, 2)),
            alert=alert,
            short_window_inference=ShortWindowInference(),
            reliability=Reliability()
        )
