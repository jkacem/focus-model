import logging

class PostureCalibrator:
    """
    Calibration des seuils personnels en début de session.
    L'étudiant regarde la caméra 5 secondes — ces valeurs
    deviennent sa référence personnelle.
    Source : principe de calibration individuelle,
    standard dans les systèmes eye-tracking (Morimoto & Mimica,
    Computer Vision and Image Understanding, 2005)
    """
    CALIBRATION_SECONDS = 5

    def __init__(self, fps: int = 15):
        self._frames = []
        self._fps = fps
        self._calibrated = False
        self.yaw_baseline   = 0.0
        self.pitch_baseline = 0.0
        self.ear_baseline   = 0.28
        self.spine_baseline = 0.0
        self.asym_baseline  = 0.0

    def feed(self, yaw: float, pitch: float, ear: float, spine_angle: float, left_sh_y: float, right_sh_y: float, sh_width: float):
        if self._calibrated:
            return
            
        asym = abs(left_sh_y - right_sh_y) / max(sh_width, 1.0)
        self._frames.append((yaw, pitch, ear, spine_angle, asym))
        
        if len(self._frames) >= self.CALIBRATION_SECONDS * self._fps:
            yaws   = [f[0] for f in self._frames]
            pitchs = [f[1] for f in self._frames]
            ears   = [f[2] for f in self._frames]
            spines = [f[3] for f in self._frames]
            asyms  = [f[4] for f in self._frames]
            
            self.yaw_baseline   = sum(yaws)   / len(yaws)
            self.pitch_baseline = sum(pitchs) / len(pitchs)
            self.ear_baseline   = sum(ears)   / len(ears)
            self.spine_baseline = sum(spines) / len(spines)
            self.asym_baseline  = sum(asyms)  / len(asyms)
            
            self._calibrated    = True
            logging.info(f"Calibration terminée — yaw_baseline={self.yaw_baseline:.2f} pitch_baseline={self.pitch_baseline:.2f} ear_baseline={self.ear_baseline:.2f} spine_baseline={self.spine_baseline:.2f}")

    @property
    def is_ready(self):
        return self._calibrated

    def reset(self):
        self._frames.clear()
        self._calibrated = False
