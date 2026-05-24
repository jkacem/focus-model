import time
from config.cv_config import config
from engine.hysteresis_manager import HysteresisManager

class StateManager:
    """
    Level 3: Sub-state Computation.
    Refactored to include persistent hysteresis with asymmetric entry/exit logic.
    """
    
    def __init__(self):
        self.hysteresis = HysteresisManager()
        self.current_states = {
            "fatigue": "normal",
            "posture": "good",
            "phone": "not_detected",
            "distraction": "focused"
        }

    def compute_sub_states(self, now: float, smoothed_scores: dict, raw_flags: dict) -> dict[str, str]:
        """
        Determines the stable sub-states using hysteresis to prevent abrupt transitions.
        """
        
        # 1. Determine Target Sub-states (Raw Observations for hysteresis)
        
        # Fatigue target (Progressive Levels)
        ema_fat = smoothed_scores["ema_fatigue"]
        if ema_fat >= config.SCORE_FATIGUE_DROWSY:
            target_fat = "drowsy"
        elif ema_fat >= config.SCORE_FATIGUE_HEAVY:
            target_fat = "fatigued"
        elif ema_fat >= config.SCORE_FATIGUE_LIGHT:
            target_fat = "slightly_fatigued"
        else:
            target_fat = "normal"
            
        # Posture target
        ema_pos = smoothed_scores["ema_posture"]
        bad_flag = raw_flags.get("bad_posture_confirmed", False)
        raw_pos = raw_flags.get("posture_raw", 100.0)
        if ema_pos >= 78:
            target_pos = "good"
        elif ema_pos < config.SCORE_POSTURE_BAD and (bad_flag or raw_pos < 60):
            target_pos = "poor_persistent"
        else:
            target_pos = "acceptable"
            
        # Phone target
        phone_sig = smoothed_scores["phone"]
        target_phone = "probable_in_use" if phone_sig > 0.45 else "not_detected"
        
        # Distraction target (Progressive Levels)
        dist_sig = smoothed_scores["distracted"]
        if dist_sig > 0.55:
            target_dist = "distracted"
        elif dist_sig > 0.25:
            target_dist = "slightly_distracted"
        else:
            target_dist = "focused"

        # Task target (Reading / Writing)
        read_sig = smoothed_scores["reading"]
        write_sig = smoothed_scores["writing"]
        if read_sig > 0.45 and read_sig > write_sig:
            target_task = "reading"
        elif write_sig > 0.45:
            target_task = "writing"
        else:
            target_task = "general"

        # 2. Apply Hysteresis (Enter vs Exit delays)
        observations = {
            "fatigue": target_fat,
            "posture": target_pos,
            "phone": target_phone,
            "distraction": target_dist,
            "task": target_task
        }
        
        self.current_states = self.hysteresis.process(now, observations)
        
        return self.current_states
