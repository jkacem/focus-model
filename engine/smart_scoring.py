import time
import numpy as np
from collections import deque
from engine.calibrator import PostureCalibrator

class ConcentrationScorer:
    ALPHA_RAPIDE = 0.20
    ALPHA_LENTE = 0.05

    def __init__(self):
        self._ema = 100.0
        self._ema_lente = 100.0
        self.yaw_history = deque(maxlen=30)

    def compute(self, yaw: float, pitch: float, ear: float, calibrator: 'PostureCalibrator') -> tuple[float, float, float]:
        self.yaw_history.append(yaw)
        
        yaw_corr = yaw - calibrator.yaw_baseline
        pitch_corr = pitch - calibrator.pitch_baseline
        ear_norm = ear / max(calibrator.ear_baseline, 0.20)
        
        gaze = max(0.0, 1.0 - abs(yaw_corr) / 35.0) * max(0.0, 1.0 - abs(pitch_corr) / 25.0)
        eye_open = max(0.0, min(1.0, (ear_norm - 0.65) / (1.0 - 0.65)))
        
        if len(self.yaw_history) > 1:
            variance = np.var(self.yaw_history)
            stability = max(0.0, 1.0 - variance / 10.0)
        else:
            stability = 1.0
            
        concentration_raw = (gaze * 0.50 + eye_open * 0.30 + stability * 0.20) * 100.0
        
        self._ema = self.ALPHA_RAPIDE * concentration_raw + (1.0 - self.ALPHA_RAPIDE) * self._ema
        self._ema_lente = self.ALPHA_LENTE * concentration_raw + (1.0 - self.ALPHA_LENTE) * self._ema_lente
        
        return self._ema, self._ema_lente, gaze

    def reset(self):
        self._ema = 100.0
        self._ema_lente = 100.0
        self.yaw_history.clear()


class PostureScorer:
    ALPHA = 0.04

    def __init__(self, fps: int = 15):
        self._ema = None
        self.bad_seconds = 0.0
        self.last_time = time.time()
        self._missing_frames = 0
        self._fps = fps

    def compute(self, spine_angle: float, left_shoulder_y: float, right_shoulder_y: float, shoulder_width: float, calibrator: 'PostureCalibrator', available: bool = True) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        if not available:
            return self.compute_fallback()

        self._missing_frames = 0

        spine_relative = max(0.0, abs(spine_angle) - calibrator.spine_baseline)
        spine_score = max(0.0, 1.0 - spine_relative / 30.0)

        if shoulder_width > 0:
            asym = abs(left_shoulder_y - right_shoulder_y) / shoulder_width
            asym_relative = max(0.0, asym - calibrator.asym_baseline)
            asym_score = max(0.0, 1.0 - asym_relative / 0.15)
        else:
            asym_score = 1.0

        p_raw = (spine_score * 0.70 + asym_score * 0.30) * 100.0
        
        if p_raw < 50.0:
            self.bad_seconds += dt
        else:
            self.bad_seconds = max(0.0, self.bad_seconds - dt * 2.0)
            
        persistence_penalty = max(0.0, min(0.40, (self.bad_seconds - 30.0) / 120.0))
        p_final = p_raw * (1.0 - persistence_penalty)

        if self._ema is None:
            self._ema = p_final
        else:
            self._ema = self.ALPHA * p_final + (1.0 - self.ALPHA) * self._ema
            
        return self._ema

    def compute_fallback(self) -> float | None:
        self._missing_frames += 1
        
        if self._ema is None:
            return None
            
        degradation = min(0.30, self._missing_frames / (3 * self._fps))
        degraded = self._ema * (1.0 - degradation)
        return max(0.0, degraded)

    def reset(self):
        self._ema = None
        self.bad_seconds = 0.0
        self._missing_frames = 0


class FatigueModulator:
    def __init__(self, fps: int = 15):
        self._f_accum = 0.0
        self._session_start = None
        self._fps = fps
        self._ear_buffer = deque(maxlen=60 * fps)

    def start_session(self):
        self._session_start = time.time()
        self._f_accum = 0.0
        self._ear_buffer.clear()

    def compute(self, ear: float) -> tuple[float, float]:
        if self._session_start is None:
            self._session_start = time.time()
            
        self._ear_buffer.append(ear)
        
        buffer_len = len(self._ear_buffer)
        if buffer_len > 0:
            closed = sum(1 for e in self._ear_buffer if e < 0.20)
            perclos = closed / buffer_len
        else:
            perclos = 0.0

        elapsed_minutes = (time.time() - self._session_start) / 60.0

        f_perclos = perclos / 0.40
        f_duration = max(0.0, min(1.0, elapsed_minutes / 120.0))
        
        fatigue_signal = f_perclos * 0.60 + f_duration * 0.40
        
        if fatigue_signal > self._f_accum:
            self._f_accum = 0.15 * fatigue_signal + 0.85 * self._f_accum
        else:
            self._f_accum = 0.01 * fatigue_signal + 0.99 * self._f_accum
            
        mu = max(0.50, min(1.00, 1.0 - 0.50 * self._f_accum))
        fatigue_out = max(0.0, min(100.0, (1.0 - mu) * 200.0))
        
        return mu, fatigue_out

    def reset(self):
        self._f_accum = 0.0
        self._session_start = None
        self._ear_buffer.clear()


class ScoreEngine:
    """
    Moteur de scoring SmartFocus — version 3.0.
    """
    def __init__(self,
                 concentration_scorer: 'ConcentrationScorer',
                 posture_scorer:   'PostureScorer',
                 fatigue_modulator:'FatigueModulator'):
        self.concentration = concentration_scorer
        self.posture       = posture_scorer
        self.fatigue       = fatigue_modulator
        
        self.total_frames = 0
        self.distracted_frames = 0
        self._vigilance_ema = 100.0
        
        self.calibrator = PostureCalibrator(fps=15)

    def compute_all(self,
                    yaw: float, pitch: float, ear: float,
                    spine_angle: float,
                    left_sh_y: float, right_sh_y: float, sh_width: float,
                    phone_detected: bool,
                    phone_confidence: float = 0.0,
                    pose_available: bool = True) -> dict:
        
        if not self.calibrator.is_ready:
            self.calibrator.feed(yaw, pitch, ear, spine_angle, left_sh_y, right_sh_y, sh_width)
            # Retourne un statut parfait pendant la calibration pour éviter des zéros
            return {
                "concentration": 100.0,
                "posture":       100.0,
                "fatigue":       0.0,
                "distraction":   0.0,
                "focus_global":  100.0,
                "posture_available": pose_available
            }

        self.total_frames += 1
        
        c, c_lente, gaze = self.concentration.compute(yaw, pitch, ear, self.calibrator)
        
        if pose_available:
            p = self.posture.compute(spine_angle, left_sh_y, right_sh_y, sh_width, self.calibrator, True)
        else:
            p = self.posture.compute_fallback()
            
        mu, fat = self.fatigue.compute(ear)
        
        buffer_len = len(self.fatigue._ear_buffer)
        if buffer_len > 0:
            closed = sum(1 for e in self.fatigue._ear_buffer if e < 0.20)
            perclos = closed / buffer_len
        else:
            perclos = 0.0
            
        vigilance_raw = max(0.0, min(1.0, 1.0 - perclos / 0.40)) * 100.0
        self._vigilance_ema = 0.08 * vigilance_raw + 0.92 * self._vigilance_ema
        
        # Distraction logic (slow EMA and relative thresholds)
        is_distracted = False
        if phone_detected and phone_confidence > 0.60:
            is_distracted = True
        elif gaze < 0.10:
            is_distracted = True
        elif c_lente < 35.0:
            is_distracted = True
            
        if is_distracted:
            self.distracted_frames += 1
            
        d_ratio = self.distracted_frames / self.total_frames if self.total_frames > 0 else 0.0
        d_ratio = max(0.0, min(0.95, d_ratio))
        distraction_out = d_ratio * 100.0

        if p is not None:
            raw_score = mu * (c * 0.55 + p * 0.25 + self._vigilance_ema * 0.20)
            posture_out = round(p, 1)
        else:
            raw_score = mu * (c * 0.80 + self._vigilance_ema * 0.20)
            posture_out = None

        focus_global = raw_score * (1.0 - d_ratio)
        focus_global = max(0.0, min(100.0, focus_global))
        focus_global = min(focus_global, c)

        return {
            "concentration": round(c, 1),
            "posture":       posture_out,
            "fatigue":       round(fat, 1),
            "distraction":   round(distraction_out, 1),
            "focus_global":  round(focus_global, 1),
            "posture_available": pose_available
        }
