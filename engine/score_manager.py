from collections import deque
import numpy as np
from config.cv_config import config

class ScoreManager:
    """
    Level 2: Score Computation & Smoothing.
    Converts raw analyzer metrics into stable smoothed signals (L2).
    """
    
    def __init__(self, fps: float, smoothing_sec: float):
        self.fps = fps
        self.smooth_len = max(1, int(smoothing_sec * fps))
        
        # Smooth signals (0..1 or 0..100)
        self._signals: dict[str, float] = {
            "reading": 0.0, "writing": 0.0, "thinking": 0.0, 
            "speech": 0.0, "social": 0.0, "phone": 0.0, "distracted": 0.0
        }
        
        # Smoothing buffers
        self._buffers: dict[str, deque] = {
            k: deque(maxlen=self.smooth_len) for k in self._signals.keys() if k != "phone"
        }
        # Ultra-fast buffer for phone
        self._buffers["phone"] = deque(maxlen=max(6, int(fps * 0.4)))

        # EMA tracks
        self.ema_fatigue = 0.0
        self.ema_posture = 100.0

    def _push_ema(self, current: float, val: float, alpha: float) -> float:
        return (alpha * val) + ((1.0 - alpha) * current)

    def _push_signal(self, key: str, value: float):
        buf = self._buffers[key]
        buf.append(max(0.0, min(1.0, float(value))))
        
        if key == "phone":
            m = float(np.percentile(buf, 90)) if buf else 0.0
            a = 0.65
        else:
            m = float(np.median(buf)) if buf else 0.0
            a = 0.15
            
        self._signals[key] = a * m + (1.0 - a) * self._signals[key]

    def compute_scores(self, raw_data: dict) -> dict[str, float]:
        """
        Takes raw L1 results and updates all internal smoothed signals.
        """
        # 1. Update work-mode signals
        self._push_signal("reading", raw_data.get("reading_ev", 0.0))
        self._push_signal("writing", raw_data.get("writing_ev", 0.0))
        self._push_signal("thinking", raw_data.get("thinking_ev", 0.0))
        self._push_signal("speech", raw_data.get("speech_ev", 0.0))
        self._push_signal("phone", raw_data.get("phone_ev", 0.0))
        self._push_signal("social", raw_data.get("social_ev", 0.0))
        self._push_signal("distracted", raw_data.get("distracted_ev", 0.0))

        # 2. Update EMA scores (Asymmetric for fatigue to ensure slow decay)
        raw_fat = raw_data.get("fatigue_sig", 0.0)
        alpha_fat = config.EMA_ALPHA_FATIGUE if raw_fat >= self.ema_fatigue else config.EMA_ALPHA_FATIGUE_DECAY
        self.ema_fatigue = self._push_ema(self.ema_fatigue, raw_fat, alpha_fat)
        self.ema_posture = self._push_ema(self.ema_posture, raw_data.get("posture_raw", 100.0), config.EMA_ALPHA_POSTURE)

        return {
            **self._signals,
            "ema_fatigue": self.ema_fatigue,
            "ema_posture": self.ema_posture
        }
