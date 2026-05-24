from collections import deque
import mediapipe as mp
import cv2
import numpy as np
from config.cv_config import config

class PostureAnalyzer:
    """
    Level 1 Analyzer: Extracts instantaneous Posture indicators.
    Returns slouch, tilt, forward head, and hand positions.
    """
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ── Calibration ───────────────────────────────────────────────────────
        self.is_calibrated       = False
        self._cal_vdist          = []
        self._cal_shoulder_width = []
        self._cal_neck_angle     = []
        self.base_vdist          = 0.15
        self.base_shoulder_width = 0.20
        self.base_neck_angle     = 0.0
        self.CAL_FRAMES          = 60
        self._bad_history        = deque(maxlen=max(1, int(config.POSTURE_BAD_HOLD_SECONDS * config.TARGET_FPS)))

    def calibrate(self, lm) -> bool:
        ls   = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs   = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = lm[self.mp_pose.PoseLandmark.NOSE]

        shoulder_mid_y = (ls.y + rs.y) / 2.0
        vdist          = shoulder_mid_y - nose.y
        sw             = abs(ls.x - rs.x)

        shoulder_mid_x = (ls.x + rs.x) / 2.0
        neck_angle     = abs(nose.x - shoulder_mid_x) / (sw + 1e-6)

        self._cal_vdist.append(vdist)
        self._cal_shoulder_width.append(sw)
        self._cal_neck_angle.append(neck_angle)

        if len(self._cal_vdist) >= self.CAL_FRAMES:
            self.base_vdist          = float(np.median(self._cal_vdist))
            self.base_shoulder_width = float(np.median(self._cal_shoulder_width))
            self.base_neck_angle     = float(np.median(self._cal_neck_angle))
            self.is_calibrated = True
            print(f"[Posture L1] Calibrated vdist={self.base_vdist:.3f} neck={self.base_neck_angle:.3f}")
            return True
        return False

    def _check_hands_on_knees(self, lm, sw: float) -> bool:
        try:
            lw = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            rw = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
            rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

            left_on = (abs(lw.y - lh.y) < config.HAND_KNEE_Y_TOL
                       and abs(lw.x - lh.x) < config.HAND_KNEE_X_TOL
                       and lw.visibility > 0.3)
            right_on = (abs(rw.y - rh.y) < config.HAND_KNEE_Y_TOL
                        and abs(rw.x - rh.x) < config.HAND_KNEE_X_TOL
                        and rw.visibility > 0.3)
            return left_on or right_on
        except Exception:
            return False

    def _check_hand_near_face(self, lm, sw: float) -> bool:
        try:
            nose = lm[self.mp_pose.PoseLandmark.NOSE]
            lw   = lm[self.mp_pose.PoseLandmark.LEFT_WRIST]
            rw   = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            nose_pt = np.array([nose.x, nose.y])

            for wrist in [lw, rw]:
                if wrist.visibility > 0.3:
                    w_pt = np.array([wrist.x, wrist.y])
                    dist = float(np.linalg.norm(w_pt - nose_pt)) / (sw + 1e-6)
                    if dist < config.HAND_FACE_DIST_TOL:
                        return True
            return False
        except Exception:
            return False

    @staticmethod
    def _midpoint(a, b) -> np.ndarray:
        return np.array([
            (a.x + b.x) / 2.0,
            (a.y + b.y) / 2.0,
            (a.z + b.z) / 2.0
        ], dtype=np.float32)

    @staticmethod
    def _deg_from_components(horizontal: float, vertical: float) -> float:
        return float(np.degrees(np.arctan2(horizontal, vertical + 1e-6)))

    def _spine_metrics(self, ls, rs, lh, rh):
        shoulder_mid = self._midpoint(ls, rs)
        hip_mid      = self._midpoint(lh, rh)
        spine_vec    = hip_mid - shoulder_mid
        forward_deg  = self._deg_from_components(spine_vec[2], spine_vec[1])
        lateral_deg  = self._deg_from_components(spine_vec[0], spine_vec[1])
        return shoulder_mid, hip_mid, forward_deg, lateral_deg

    def _classify_inclination(self, forward_deg: float, lateral_deg: float):
        if abs(forward_deg) > config.POSTURE_FORWARD_DEG_TOL:
            axis = "forward" if forward_deg > 0 else "backward"
            return axis, abs(forward_deg)
        if abs(lateral_deg) > config.POSTURE_LATERAL_DEG_TOL:
            axis = "right" if lateral_deg > 0 else "left"
            return axis, abs(lateral_deg)
        return "neutral", 0.0

    def _update_bad_history(self, flag: bool) -> bool:
        if self._bad_history.maxlen == 0:
            return flag
        self._bad_history.append(1 if flag else 0)
        if len(self._bad_history) < self._bad_history.maxlen:
            return False
        ratio = sum(self._bad_history) / len(self._bad_history)
        return ratio > 0.6

    def analyze(self, image, calibrating: bool = False) -> dict:
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return {
                "slouch_score": 0.0,
                "tilt_score": 0.0,
                "fwd_score": 0.0,
                "lean_score": 0.0,
                "hands_on_knees": False,
                "hand_near_face": False,
                "is_calibrated": self.is_calibrated,
                "inclination_axis": "neutral",
                "inclination_degrees": 0.0,
                "forward_inclination_deg": 0.0,
                "lateral_inclination_deg": 0.0,
                "posture_score": 0.0,
                "bad_posture_confirmed": False,
                "left_shoulder_y": 0.0,
                "right_shoulder_y": 0.0,
                "shoulder_width": 0.0,
                "spine_angle": 0.0,
                "pose_available": False
            }

        lm = result.pose_landmarks.landmark

        if calibrating:
            self.calibrate(lm)
            return {"calibrating": True, "progress": len(self._cal_vdist) / self.CAL_FRAMES}

        ls   = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs   = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh   = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh   = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]
        nose = lm[self.mp_pose.PoseLandmark.NOSE]
        sw   = abs(ls.x - rs.x)

        # ── Visibility guard ───────────────────────────────────────────────────
        # When head is turned far enough that shoulders become occluded, spine
        # metrics become unreliable.  Return the last confirmed EMA values as
        # a neutral reading rather than emitting a wildly wrong score.
        avg_shoulder_vis = (ls.visibility + rs.visibility) / 2.0
        if avg_shoulder_vis < 0.40:
            return {
                "slouch_score": 0.7,
                "tilt_score": 0.8,
                "fwd_score": 0.8,
                "lean_score": 0.8,
                "hands_on_knees": False,
                "hand_near_face": False,
                "is_calibrated": self.is_calibrated,
                "inclination_axis": "neutral",
                "inclination_degrees": 0.0,
                "forward_inclination_deg": 0.0,
                "lateral_inclination_deg": 0.0,
                "posture_score": 70.0,
                "bad_posture_confirmed": False,
                "left_shoulder_y": ls.y,
                "right_shoulder_y": rs.y,
                "shoulder_width": sw,
                "spine_angle": 0.0,
                "pose_available": False
            }

        shoulder_mid, hip_mid, forward_deg, lateral_deg = self._spine_metrics(ls, rs, lh, rh)
        axis_label, axis_severity = self._classify_inclination(forward_deg, lateral_deg)

        shoulder_mid_y = shoulder_mid[1]
        vdist          = shoulder_mid_y - nose.y
        slouch_ratio   = min(1.0, max(0.0, vdist / (self.base_vdist + 1e-6)))
        forward_factor = max(0.0, 1.0 - abs(forward_deg) / config.POSTURE_FORWARD_DEG_MAX)
        slouch_score   = float(np.clip(slouch_ratio * forward_factor, 0.0, 1.0))

        tilt   = abs(ls.y - rs.y)
        tilt_n = tilt / (sw + 1e-6)
        tilt_score = max(0.0, 1.0 - tilt_n * config.TILT_PENALTY_MULT)
        lateral_factor = max(0.0, 1.0 - abs(lateral_deg) / config.POSTURE_LATERAL_DEG_MAX)
        tilt_score = float(np.clip(tilt_score * lateral_factor, 0.0, 1.0))

        shoulder_mid_x = shoulder_mid[0]
        fwd_head       = abs(nose.x - shoulder_mid_x) / (sw + 1e-6)
        fwd_deviation  = max(0.0, fwd_head - self.base_neck_angle)
        fwd_score      = max(0.0, 1.0 - fwd_deviation * config.FWD_HEAD_PENALTY_MULT)
        fwd_score      = float(np.clip(fwd_score * forward_factor, 0.0, 1.0))

        lean_offset = abs(shoulder_mid[0] - hip_mid[0]) / (sw + 1e-6)
        lean_score  = max(0.0, 1.0 - lean_offset * config.LEAN_PENALTY_MULT)
        lean_score  = float(np.clip(lean_score * lateral_factor, 0.0, 1.0))

        raw_hands_knee = self._check_hands_on_knees(lm, sw)
        raw_hand_face  = self._check_hand_near_face(lm, sw)

        posture_issue_flag = (
            axis_label != "neutral"
            or slouch_score < 0.40
            or tilt_score < 0.40
            or lean_score < 0.40
        )
        bad_posture_confirmed = self._update_bad_history(posture_issue_flag)

        posture_score = (
            0.40 * slouch_score +
            0.20 * fwd_score +
            0.15 * tilt_score +
            0.25 * lean_score
        ) * 100.0
        if axis_label != "neutral":
            posture_score -= min(12.0, axis_severity / max(config.POSTURE_FORWARD_DEG_MAX, 1.0) * 8.0)
        posture_score = float(np.clip(posture_score, 0.0, 100.0))

        inclination_deg = float(max(abs(forward_deg), abs(lateral_deg)))

        return {
            "slouch_score": float(slouch_score),
            "tilt_score": float(tilt_score),
            "fwd_score": float(fwd_score),
            "lean_score": float(lean_score),
            "hands_on_knees": raw_hands_knee,
            "hand_near_face": raw_hand_face,
            "is_calibrated": self.is_calibrated,
            "inclination_axis": axis_label,
            "inclination_degrees": round(inclination_deg, 2),
            "forward_inclination_deg": round(forward_deg, 2),
            "lateral_inclination_deg": round(lateral_deg, 2),
            "posture_score": round(posture_score, 1),
            "bad_posture_confirmed": bad_posture_confirmed,
            "left_shoulder_y": float(ls.y),
            "right_shoulder_y": float(rs.y),
            "shoulder_width": float(sw),
            "spine_angle": round(inclination_deg, 2),
            "pose_available": True
        }
