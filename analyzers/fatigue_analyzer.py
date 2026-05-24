"""
FatigueAnalyzer - Level 1 Analyzer
Corrections apportées :
  1. Compensation EAR selon yaw (évite faux positifs quand tête tournée)
  2. Séparation fatigue/stress : le jitter lent n'est plus compté comme stress
  3. Seuil PERCLOS plus conservateur pour éviter faux positifs
"""

import time
from collections import deque
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import cv2
import numpy as np
from config.cv_config import config

# ── Eye landmark indices (MediaPipe FaceMesh) ─────────────────────────────────
LEFT_EYE  = [33,  160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ── Mouth landmarks ───────────────────────────────────────────────────────────
MOUTH_TOP_CENTER    = 13
MOUTH_BOTTOM_CENTER = 14
MOUTH_TOP_LEFT      = 81
MOUTH_BOTTOM_LEFT   = 178
MOUTH_TOP_RIGHT     = 311
MOUTH_BOTTOM_RIGHT  = 402
MOUTH_LEFT_CORNER   = 78
MOUTH_RIGHT_CORNER  = 308

# ── Compensation yaw ──────────────────────────────────────────────────────────
# Au-delà de ce seuil, l'EAR baisse mécaniquement → on relaxe le seuil
YAW_COMPENSATION_START_DEG = 12.0   # début de la compensation
YAW_COMPENSATION_MAX_DEG   = 35.0   # compensation maximale
EAR_YAW_RELAX_MAX          = 0.18   # relaxation max (18%) au yaw maximum
YAWN_MOUTH_RATIO_MIN       = 1.40  # mouth must open 40% above baseline (vs 0.75 ratio vs MAR) to count as yawn
YAWN_FRAMES_MIN            = 22    # ~0.7s at effective FPS; filters talking bursts


class FatigueAnalyzer:
    """
    Level 1 Analyzer: Signaux instantanés de fatigue.
    Accepte un yaw optionnel (depuis AttentionAnalyzer) pour corriger l'EAR.
    """

    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ── Calibration ───────────────────────────────────────────────────────
        self.is_calibrated     = False
        self.base_ear          = 0.25
        self.ear_threshold     = 0.21
        self.base_mar          = 0.0
        self.base_pitch_proxy  = 0.0
        self._cal_ear_data     = []
        self._cal_mar_data     = []
        self._cal_pitch_data   = []
        self.CAL_FRAMES        = 60

        self._effective_fps    = max(1.0, config.TARGET_FPS / 2.0)
        self._perclos_window   = deque(maxlen=max(1, int(config.PERCLOS_WINDOW_SECONDS * self._effective_fps)))
        self._eye_closed_start = None
        self._yawn_active      = False
        self._yawn_start       = 0.0
        self._yawn_counter     = 0
        self._last_yawn_time   = 0.0
        self._yawn_history     = deque()
        self._yawn_count_total = 0

    # ── Calibration ──────────────────────────────────────────────────────────
    def calibrate(self, ear: float, mar: float, pitch_proxy: float) -> bool:
        self._cal_ear_data.append(ear)
        self._cal_mar_data.append(mar)
        self._cal_pitch_data.append(pitch_proxy)
        if len(self._cal_ear_data) >= self.CAL_FRAMES:
            self.base_ear      = float(np.median(self._cal_ear_data))
            self.ear_threshold = self.base_ear * 0.75  # MORE SENSITIVE (was 0.62)
            self.base_mar      = float(np.median(self._cal_mar_data))
            if self._cal_pitch_data:
                self.base_pitch_proxy = float(np.median(self._cal_pitch_data))
            self.is_calibrated = True
            print(f"[Fatigue L1] Calibrated base_ear={self.base_ear:.3f}  thresh={self.ear_threshold:.3f}  base_mar={self.base_mar:.3f}")
            return True
        return False

    # ── EAR / MAR / pitch ────────────────────────────────────────────────────
    @staticmethod
    def _ear(landmarks, indices) -> float:
        try:
            p = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
            v1 = np.linalg.norm(p[1] - p[5])
            v2 = np.linalg.norm(p[2] - p[4])
            h  = np.linalg.norm(p[0] - p[3])
            return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.25
        except Exception:
            return 0.25

    @staticmethod
    def _mar(landmarks) -> float:
        try:
            tc = np.array([landmarks[MOUTH_TOP_CENTER].x,    landmarks[MOUTH_TOP_CENTER].y])
            bc = np.array([landmarks[MOUTH_BOTTOM_CENTER].x, landmarks[MOUTH_BOTTOM_CENTER].y])
            tl = np.array([landmarks[MOUTH_TOP_LEFT].x,      landmarks[MOUTH_TOP_LEFT].y])
            bl = np.array([landmarks[MOUTH_BOTTOM_LEFT].x,   landmarks[MOUTH_BOTTOM_LEFT].y])
            tr = np.array([landmarks[MOUTH_TOP_RIGHT].x,     landmarks[MOUTH_TOP_RIGHT].y])
            br = np.array([landmarks[MOUTH_BOTTOM_RIGHT].x,  landmarks[MOUTH_BOTTOM_RIGHT].y])
            lc = np.array([landmarks[MOUTH_LEFT_CORNER].x,   landmarks[MOUTH_LEFT_CORNER].y])
            rc = np.array([landmarks[MOUTH_RIGHT_CORNER].x,  landmarks[MOUTH_RIGHT_CORNER].y])
            v1 = np.linalg.norm(tc - bc)
            v2 = np.linalg.norm(tl - bl)
            v3 = np.linalg.norm(tr - br)
            h  = np.linalg.norm(lc - rc)
            return float((v1 + v2 + v3) / (3.0 * h)) if h > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _pitch_proxy(landmarks) -> float:
        try:
            nose  = np.array([landmarks[1].x,   landmarks[1].y])
            l_eye = np.array([landmarks[33].x,  landmarks[33].y])
            r_eye = np.array([landmarks[263].x, landmarks[263].y])
            return float(nose[1] - (l_eye[1] + r_eye[1]) / 2.0)
        except Exception:
            return 0.0

    # ── FIX #1 : Compensation EAR selon le yaw ───────────────────────────────
    def _yaw_compensated_threshold(self, base_thresh: float, yaw_deg: float) -> float:
        """
        Quand la tête tourne, l'œil visible se projette différemment → EAR
        diminue mécaniquement. On relaxe le seuil de fermeture pour ne pas
        déclencher de fausse fatigue.
        """
        abs_yaw = abs(yaw_deg)
        if abs_yaw <= YAW_COMPENSATION_START_DEG:
            return base_thresh

        # Interpolation linéaire entre 0 et EAR_YAW_RELAX_MAX
        t = min(1.0, (abs_yaw - YAW_COMPENSATION_START_DEG) /
                     (YAW_COMPENSATION_MAX_DEG - YAW_COMPENSATION_START_DEG))
        relax_factor = 1.0 - EAR_YAW_RELAX_MAX * t
        return base_thresh * relax_factor

    # ── Main analyze ─────────────────────────────────────────────────────────
    def analyze(self, image, calibrating: bool = False, yaw_deg: float = 0.0) -> dict:
        """
        Paramètre supplémentaire : yaw_deg (depuis AttentionAnalyzer).
        Permet la compensation EAR quand la tête est tournée.
        """
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        _empty = {
            "ear": 0.0, "mar": 0.0, "pitch_proxy": 0.0,
            "eye_closed": False, "perclos": 0.0,
            "fatigue_score": 0.0, "fatigue_level": "low",
            "microsleep": False, "slow_blink": False,
            "yawn_in_progress": False, "yawn_count": self._yawn_count_total,
            "frequent_yawn": False, "yawn_frequency_per_min": 0.0,
            "reading_head_down": False, "reading_compensation": False,
            "ear_threshold_active": round(self.ear_threshold, 3),
            "base_mar": round(self.base_mar, 3),
            "is_calibrated": self.is_calibrated
        }

        if not result.multi_face_landmarks:
            return _empty

        lm      = result.multi_face_landmarks[0].landmark
        avg_ear = (self._ear(lm, LEFT_EYE) + self._ear(lm, RIGHT_EYE)) / 2.0
        mar         = self._mar(lm)
        pitch_proxy = self._pitch_proxy(lm)

        if calibrating:
            self.calibrate(avg_ear, mar, pitch_proxy)
            return {"calibrating": True, "progress": len(self._cal_ear_data) / self.CAL_FRAMES}

        now = time.time()

        # ── Compensation pitch (lecture) ──────────────────────────────────────
        pitch_delta        = pitch_proxy - self.base_pitch_proxy
        reading_head_down  = pitch_delta > config.READING_PITCH_DELTA
        effective_thresh   = self.ear_threshold
        reading_comp_applied = False
        if reading_head_down:
            effective_thresh   *= (1.0 - config.READING_EAR_RELAXATION)
            reading_comp_applied = True

        # ── FIX #1 : Compensation yaw ─────────────────────────────────────────
        effective_thresh = self._yaw_compensated_threshold(effective_thresh, yaw_deg)

        eye_closed     = avg_ear < effective_thresh
        sleepy_eye     = avg_ear < (self.base_ear * (1.0 - config.EYE_DROWSY_RELAXATION))
        eye_closure_deep = avg_ear < (effective_thresh * 0.92)

        # ── PERCLOS ───────────────────────────────────────────────────────────
        if self._perclos_window.maxlen:
            self._perclos_window.append(1 if eye_closed else 0)
            perclos = (sum(self._perclos_window) / len(self._perclos_window)) * 100.0
        else:
            perclos = 0.0

        # ── Slow blink / microsleep ───────────────────────────────────────────
        slow_blink = False
        if eye_closed:
            if self._eye_closed_start is None:
                self._eye_closed_start = now
        else:
            if self._eye_closed_start is not None:
                blink_dur  = now - self._eye_closed_start
                slow_blink = blink_dur >= config.SLOW_BLINK_SECONDS
                self._eye_closed_start = None

        microsleep = False
        if self._eye_closed_start is not None:
            microsleep = (now - self._eye_closed_start) >= config.MICROSLEEP_SECONDS

        # ── Yawn ─────────────────────────────────────────────────────────────
        yawn_threshold = max(self.base_mar + config.YAWN_MAR_BASE_OFFSET,
                             self.base_mar * 1.6)
        mouth_ratio = mar / max(self.base_mar, 1e-3)
        yawn_candidate = (mar > yawn_threshold) and (mouth_ratio > YAWN_MOUTH_RATIO_MIN)
        if yawn_candidate:
            self._yawn_counter += 1
            if self._yawn_counter > YAWN_FRAMES_MIN and not self._yawn_active:
                self._yawn_active = True
                self._yawn_start  = now
        else:
            if self._yawn_active:
                duration = now - self._yawn_start
                if (duration >= config.YAWN_SUSTAIN_SECONDS and
                        (now - self._last_yawn_time) >= config.YAWN_COOLDOWN_SECONDS):
                    self._yawn_count_total += 1
                    self._yawn_history.append(now)
                    self._last_yawn_time = now
                self._yawn_active = False
            self._yawn_counter = 0

        while self._yawn_history and (now - self._yawn_history[0]) > config.YAWN_FREQ_WINDOW_SEC:
            self._yawn_history.popleft()

        frequent_yawn    = len(self._yawn_history) >= config.YAWN_FREQ_MIN_COUNT
        yawn_frequency   = (len(self._yawn_history) / max(1.0, config.YAWN_FREQ_WINDOW_SEC)) * 60.0

        # ── Fatigue score ────────────────────────────────────────────────────
        fatigue_score = perclos * 0.8
        if slow_blink:
            fatigue_score += 8.0
        if self._yawn_history:
            fatigue_score += min(18.0, len(self._yawn_history) * config.YAWN_FATIGUE_BONUS)
        if frequent_yawn:
            fatigue_score += 10.0
        if sleepy_eye:
            fatigue_score += config.SLEEPY_EAR_BONUS
        if eye_closure_deep:
            fatigue_score += 6.0
        if eye_closed:
            fatigue_score += 8.0   # Reduced: avoid instant spike (was 15)
        if self._yawn_active:
            fatigue_score += 10.0  # Reduced: in-progress yawn is a weak signal (was 25)
        if microsleep:
            fatigue_score = max(fatigue_score, 90.0)
        fatigue_score = float(np.clip(fatigue_score, 0.0, 100.0))

        if microsleep or fatigue_score >= config.SCORE_FATIGUE_HIGH:
            fatigue_level = "high"
        elif fatigue_score >= config.SCORE_FATIGUE_WARNING:
            fatigue_level = "moderate"
        else:
            fatigue_level = "low"

        return {
            "ear":                   round(avg_ear, 3),
            "mar":                   round(mar, 3),
            "pitch_proxy":           round(pitch_proxy, 4),
            "eye_closed":            eye_closed,
            "perclos":               round(perclos, 1),
            "fatigue_score":         round(fatigue_score, 1),
            "fatigue_level":         fatigue_level,
            "microsleep":            microsleep,
            "slow_blink":            slow_blink,
            "yawn_in_progress":      self._yawn_active,
            "yawn_count":            self._yawn_count_total,
            "frequent_yawn":         frequent_yawn,
            "yawn_frequency_per_min": round(yawn_frequency, 2),
            "reading_head_down":     reading_head_down,
            "reading_compensation":  reading_comp_applied,
            "sleepy_eye":            sleepy_eye,
            "ear_threshold_active":  round(effective_thresh, 3),
            "base_mar":              round(self.base_mar, 3),
            "is_calibrated":         self.is_calibrated
        }
