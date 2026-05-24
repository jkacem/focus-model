import mediapipe as mp 
import cv2 
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip        – idx 1
    (0.0, -330.0, -65.0),     # Chin             – idx 152
    (-225.0, 170.0, -135.0),  # Left eye corner  – idx 33
    (225.0, 170.0, -135.0),   # Right eye corner – idx 263
    (-150.0, -150.0, -125.0), # Left mouth       – idx 61
    (150.0, -150.0, -125.0),  # Right mouth      – idx 291
], dtype=np.float64)

LANDMARK_IDS = [1, 152, 33, 263, 61, 291]
SOCIAL_GAZE_YAW_MIN = 10.0   # minimum yaw degrees to be considered looking sideways

class AttentionAnalyzer:
    """
    Level 1 Analyzer: Extracts purely instantaneous observations for Attention.
    Does not hold temporal buffers or maintain state, except for calibration baselines.
    """
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # ── Calibration ───────────────────────────────────────────────────────
        self.is_calibrated   = False
        self.base_yaw        = 0.0
        self.base_pitch      = 0.0
        self._cal_yaws       = []
        self._cal_pitches    = []
        self.CAL_FRAMES      = 60

        # ── Camera matrix ─────────────────────────────────────────────────────
        self._cam_matrix  = None
        self._dist_coeffs = None

    def calibrate(self, yaw: float, pitch: float) -> bool:
        self._cal_yaws.append(yaw)
        self._cal_pitches.append(pitch)
        if len(self._cal_yaws) >= self.CAL_FRAMES:
            self.base_yaw   = float(np.median(self._cal_yaws))
            self.base_pitch = float(np.median(self._cal_pitches))
            self.is_calibrated = True
            print(f"[Attention L1] Calibrated yaw={self.base_yaw:.1f}° pitch={self.base_pitch:.1f}°")
            return True
        return False

    def _head_pose(self, landmarks, img_w: int, img_h: int):
        if self._cam_matrix is None:
            focal = img_w
            cx, cy = img_w / 2, img_h / 2
            self._cam_matrix = np.array([
                [focal, 0,  cx],
                [0, focal,  cy],
                [0, 0,       1]
            ], dtype=np.float64)
            self._dist_coeffs = np.zeros((4, 1))

        image_points = np.array([
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
            for idx in LANDMARK_IDS
        ], dtype=np.float64)

        success, rvec, _ = cv2.solvePnP(
            MODEL_POINTS, image_points,
            self._cam_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(angles[1]), float(angles[0])   # yaw, pitch

    @staticmethod
    def _mouth_aspect_ratio(landmarks) -> float:
        try:
            top    = np.array([landmarks[13].x, landmarks[13].y])
            bottom = np.array([landmarks[14].x, landmarks[14].y])
            left   = np.array([landmarks[61].x, landmarks[61].y])
            right  = np.array([landmarks[291].x, landmarks[291].y])
            vert   = np.linalg.norm(top - bottom)
            horiz  = np.linalg.norm(left - right) + 1e-6
            return float(vert / horiz)
        except Exception:
            return 0.0

    @staticmethod
    def _face_center_x(landmarks) -> float:
        try:
            nose_x = landmarks[1].x
            return float(nose_x)
        except Exception:
            return 0.5

    @staticmethod
    def _face_bbox(landmarks, img_w: int, img_h: int):
        try:
            xs = [lm.x for lm in landmarks]
            ys = [lm.y for lm in landmarks]
            x1 = max(0, int(min(xs) * img_w))
            y1 = max(0, int(min(ys) * img_h))
            x2 = min(img_w - 1, int(max(xs) * img_w))
            y2 = min(img_h - 1, int(max(ys) * img_h))
            return (x1, y1, x2, y2)
        except Exception:
            return None

    def analyze(self, image, calibrating: bool = False) -> dict:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        num_faces = len(result.multi_face_landmarks) if result.multi_face_landmarks else 0

        if num_faces == 0:
            return {
                "face_present": False,
                "num_faces": 0,
                "yaw": 0.0,
                "pitch": 0.0,
                "mar": 0.0,
                "face_bbox": None,
                "gaze_toward_person": False,
                "is_calibrated": self.is_calibrated
            }

        primary = result.multi_face_landmarks[0].landmark
        face_bbox = self._face_bbox(primary, w, h)
        yaw_raw, pitch_raw = self._head_pose(primary, w, h)
        mar = self._mouth_aspect_ratio(primary)

        if calibrating:
            self.calibrate(yaw_raw, pitch_raw)
            return {"calibrating": True, "progress": len(self._cal_yaws) / self.CAL_FRAMES}

        # Calibrated relative angles
        yaw = yaw_raw - self.base_yaw
        pitch = pitch_raw - self.base_pitch

        # Social gaze indicator (instantaneous)
        gaze_toward_person = False
        if num_faces >= 2:
            primary_x = self._face_center_x(primary)
            second_lm = result.multi_face_landmarks[1].landmark
            second_x = self._face_center_x(second_lm)
            
            face_is_right = second_x > primary_x
            looking_right = yaw > SOCIAL_GAZE_YAW_MIN
            looking_left  = yaw < -SOCIAL_GAZE_YAW_MIN
            
            gaze_toward_person = (
                (face_is_right and looking_right) or
                (not face_is_right and looking_left)
            )

        return {
            "face_present": True,
            "num_faces": num_faces,
            "yaw": round(yaw, 2),
            "pitch": round(pitch, 2),
            "mar": round(mar, 3),
            "face_bbox": face_bbox,
            "gaze_toward_person": gaze_toward_person,
            "is_calibrated": self.is_calibrated
        }
