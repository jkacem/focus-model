import cv2

class MinimalUI:
    """Minimal, professional UI overlay for production use."""

    def __init__(self, window_name: str = "SmartFocus", debug: bool = False):
        self.window_name = window_name
        self.debug = debug
        self._opened = False

    def open(self, width: int = 800, height: int = 600):
        if self._opened:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)
        self._opened = True

    def close(self):
        if not self._opened:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self._opened = False

    @staticmethod
    def _state_color(state: str) -> tuple[int, int, int]:
        s = (state or "").lower()
        if "distraction" in s or "off_task" in s:
            return (0, 0, 255) # Red for alert
        if "focused" in s:
            return (0, 200, 80) # Clean green
        if "thinking" in s or "explaining" in s or "reading" in s or "writing" in s:
            return (220, 180, 100) # Soft blue/gold
        return (255, 255, 255)

    @staticmethod
    def _factor_color(kind: str, value: str) -> tuple[int, int, int]:
        v = (value or "").lower()
        if kind == "posture":
            if "poor" in v or "bad" in v: return (0, 0, 255)
            if "acceptable" in v or "warning" in v: return (0, 165, 255)
            return (0, 200, 80)
        if kind == "fatigue":
            if "high" in v: return (0, 0, 255)
            if "medium" in v or "warning" in v: return (0, 165, 255)
            return (0, 200, 80)
        return (220, 220, 220)

    def draw(
        self,
        frame,
        *,
        state: str,
        confidence: float,
        phone_detected: bool,
        factors: dict | None = None,
        scores: dict | None = None,
        calibrating: bool = False,
        calibration_progress: float | None = None,
        reasoning: list[str] | None = None
    ):
        h, w = frame.shape[:2]
        bar_h = 100 

        # Translations dictionary in English to prevent any encoding/accent rendering issues in OpenCV
        TRANS = {
            "focused": "FOCUSED", "focused_reading": "READING", "focused_writing": "WRITING",
            "thinking": "THINKING", "self_explaining": "SELF-EXPLAINING",
            "brief_off_task": "DISTRACTED", "phone_distraction": "DISTRACTION (PHONE)",
            "social_distraction": "DISTRACTION (SOCIAL)",
            "good": "Good", "warning": "Warning", "bad": "Bad",
            "normal": "Normal", "fatigue_warning": "Warning", "fatigue_high": "Critical",
            "alone": "Alone", "other_person_present": "Presence", "active_interaction": "Interaction",
            "not_detected": "Not Detected", "detected_not_used": "Detected", "probable_in_use": "In Hand",
            "acceptable": "Acceptable", "poor_persistent": "Bad (Persistent)"
        }

        # High-quality dark header bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        if calibrating:
            pct = 0.0 if calibration_progress is None else max(0.0, min(1.0, float(calibration_progress)))
            cv2.rectangle(frame, (0, bar_h - 5), (int(w * pct), bar_h), (0, 220, 100), -1)
            cv2.putText(frame, "INITIALIZING SYSTEM...", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return

        # 1. Main Mode Header
        mode_text = TRANS.get(state.lower(), state.upper())
        state_color = self._state_color(state)
        cv2.putText(frame, f"MODE: {mode_text}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, state_color, 2)
        
        # 2. Production indicators (Top Bar)
        if factors:
            spacing = w // 5
            x_offset = 15
            y_label = 65
            y_val = 88

            # Mapping factors to the UI items in English
            ui_items = [
                ("Posture", TRANS.get(factors.get("posture_state", "good").lower(), factors.get("posture_state", "???")), "posture"),
                ("Fatigue", TRANS.get(factors.get("fatigue_state", "normal").lower(), "Normal"), "fatigue"),
                ("Social", TRANS.get(factors.get("social_state", "alone").lower(), "Alone"), "social"),
                ("Phone", "Detected" if phone_detected else "Not Detected", "phone"),
            ]

            # Instant overrides for UI responsiveness (as in previous turn)
            if factors.get("eye_closed_instant"):
                ui_items[1] = ("Fatigue", "DROWSINESS", "fatigue")
            if factors.get("yawn_instant"):
                ui_items[1] = ("Fatigue", "YAWNING", "fatigue")

            for i, (lbl, val, kind) in enumerate(ui_items):
                x = x_offset + i * spacing
                if x + 100 > w: break
                
                # Determine color based on value text or severity (English words)
                lower_val = val.lower()
                if any(k in lower_val for k in ["bad", "critical", "drowsiness", "yawning", "detected", "interaction", "poor"]):
                    color = (0, 0, 255) # Red
                elif any(k in lower_val for k in ["warning", "acceptable", "suspected"]):
                    color = (0, 165, 255) # Orange
                else:
                    color = (0, 200, 80) # Green

                cv2.putText(frame, lbl, (x, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                cv2.putText(frame, val, (x, y_val), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 3. Reasoning Panel (Analyse) - Bottom Right
        if reasoning and len(reasoning) > 0:
            rx, ry = w - 280, h - 140
            panel_overlay = frame.copy()
            cv2.rectangle(panel_overlay, (rx, ry), (w - 10, h - 10), (30, 30, 30), -1)
            cv2.addWeighted(panel_overlay, 0.6, frame, 0.4, 0, frame)
            
            cv2.putText(frame, "Analysis:", (rx + 10, ry + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            for i, reason in enumerate(reasoning[:4]):
                clean_reason = reason.replace("_", " ").capitalize()
                cv2.putText(frame, f"- {clean_reason}", (rx + 15, ry + 50 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


    def show(self, frame) -> bool:
        if not self._opened:
            self.open()
        cv2.imshow(self.window_name, frame)
        return (cv2.waitKey(1) & 0xFF) == ord("q")
