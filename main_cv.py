from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import cv2

from analyzers.attention_analyzer import AttentionAnalyzer
from analyzers.fatigue_analyzer import FatigueAnalyzer
from analyzers.phone_detector import PhoneDetector
from analyzers.posture_analyzer import PostureAnalyzer
from engine.session_tracker import SessionTracker
from engine.temporal_engine import TemporalEngine
from engine.score_smoother import ScoreSmoother
from output.api_client import APIClient
from output.json_formatter import JSONFormatter
from engine.smart_scoring import ScoreEngine, ConcentrationScorer, PostureScorer, FatigueModulator
from ui.minimal_ui import MinimalUI


# --- Performance / cadence ---
POSTURE_SKIP = 10   # FASTER
PHONE_SKIP   = 8    # FASTER (was 12)
CAL_SECONDS  = 3


def open_camera(index: int):
    backends: list[int] = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    backends.append(0)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
        cap.release()

    return cv2.VideoCapture(index)


class SmartFocusPipelineV3:
    """Detection → temporal processing → decisions → JSON output → optional UI."""

    def __init__(
        self,
        *,
        ui: bool = False,
        ui_debug: bool = False,
        send_to_backend: bool = True,
        duration: int = 0,
        output_json: bool = True,
        output_interval_sec: float = 0.5,
        camera_index: int = 0,
        backend_url: str = "http://localhost:8000/api/v1",
    ):
        self.session_id = str(uuid.uuid4())
        self.duration = int(duration)
        self.send_to_backend = bool(send_to_backend)
        self.output_json = bool(output_json)
        self.output_interval_sec = float(output_interval_sec)

        # Detection (L1)
        self.attention = AttentionAnalyzer()
        self.fatigue = FatigueAnalyzer()
        self.posture = PostureAnalyzer()
        self.phone = PhoneDetector()
        # StressAnalyzer removed per user request

        # Temporal + decision engine (L2/L3/L4)
        self.engine = TemporalEngine(self.session_id)

        # Output
        self.conc_scorer = ConcentrationScorer()
        self.pos_scorer = PostureScorer()
        self.fat_mod = FatigueModulator()
        self.score_engine = ScoreEngine(self.conc_scorer, self.pos_scorer, self.fat_mod)
        self.formatter = JSONFormatter()
        self.api_client = APIClient(base_url=backend_url)
        self.tracker = SessionTracker(session_id=self.session_id)

        # Smoothers for L1 outputs (latency=5 frames)
        self.att_smoother = ScoreSmoother(window_size=5)
        self.fat_smoother = ScoreSmoother(window_size=5)
        self.pos_smoother = ScoreSmoother(window_size=5)

        # Optional UI (minimal; no raw intermediate states)
        self.ui = MinimalUI(window_name="SmartFocus", debug=ui_debug) if ui else None

        # Camera
        self.cap = open_camera(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Runtime
        self._start = time.time()
        self._runtime_start: float | None = None
        self._frame = 0

        self._last_posture: dict = {}
        self._last_phone: dict = {"phone_found": False}
        self._last_att: dict = {}
        self._last_fat: dict = {}
        self._last_str: dict = {}

        self._last_emit = 0.0
        self._last_ui_scores: dict | None = None

        # Local JSON output
        self._frames_fp = None
        self._frames_path: Path | None = None
        self._summary_path: Path | None = None

    def _is_calibrating(self) -> bool:
        return (time.time() - self._start) < CAL_SECONDS

    def _open_output_files(self):
        if not self.output_json:
            return

        out_dir = (Path(__file__).resolve().parent / "output" / "sessions")
        out_dir.mkdir(parents=True, exist_ok=True)

        self._frames_path = out_dir / f"{self.session_id}.jsonl"
        self._summary_path = out_dir / f"{self.session_id}_summary.json"
        self._frames_fp = open(self._frames_path, "w", encoding="utf-8")

    def _close_output_files(self):
        if self._frames_fp is not None:
            try:
                self._frames_fp.close()
            except Exception:
                pass
            self._frames_fp = None

    def run(self):
        if self.send_to_backend:
            self.api_client.ensure_session(self.session_id)

        self._open_output_files()

        self.fat_mod.start_session()

        if self.ui:
            self.ui.open(width=960, height=540)

        print(f"[SmartFocus] Session {self.session_id} started. Calibrating for {CAL_SECONDS}s...")

        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # Auto-stop only after calibration
                if (
                    self._runtime_start is not None
                    and self.duration > 0
                    and (time.time() - self._runtime_start) >= self.duration
                ):
                    print(f"[SmartFocus] Duration of {self.duration}s reached. Stopping.")
                    break

                self._frame += 1
                now = time.time()
                elapsed = now - self._start
                calibrating = self._is_calibrating()

                analysis_frame = None
                if (self._frame % 4 == 0) or (self._frame % POSTURE_SKIP == 0) or (self._frame % PHONE_SKIP == 0):
                    analysis_frame = cv2.resize(frame, (160, 120))

                # L1 analyzers (skipped for performance)
                if self._frame % 4 == 0:
                    self._last_att = self.attention.analyze(analysis_frame, calibrating=calibrating)
                    yaw_deg = self._last_att.get("yaw", 0.0) if not calibrating else 0.0
                    self._last_fat = self.fatigue.analyze(analysis_frame, calibrating=calibrating, yaw_deg=yaw_deg)

                if self._frame % POSTURE_SKIP == 0:
                    self._last_posture = self.posture.analyze(analysis_frame, calibrating=calibrating)

                if self._frame % PHONE_SKIP == 0:
                    face_box = self._last_att.get("face_bbox")
                    if face_box is not None:
                        src_h, src_w = 120, 160  # attention analyzer input size
                        dst_h, dst_w = frame.shape[:2]
                        sx = dst_w / max(1, src_w)
                        sy = dst_h / max(1, src_h)
                        x1, y1, x2, y2 = face_box
                        face_box = (
                            int(x1 * sx),
                            int(y1 * sy),
                            int(x2 * sx),
                            int(y2 * sy),
                        )
                    self._last_phone = self.phone.analyze(frame, face_box=face_box)

                att_result = self._last_att or {}
                fat_result = self._last_fat or {}

                if calibrating:
                    if self.ui:
                        self.ui.draw(
                            frame,
                            state="focused",
                            confidence=0.0,
                            phone_detected=False,
                            calibrating=True,
                            calibration_progress=min(1.0, elapsed / CAL_SECONDS),
                        )
                        if self.ui.show(frame):
                            break
                    continue

                if self._runtime_start is None:
                    self._runtime_start = time.time()
                    self._last_emit = 0.0

                yaw = att_result.get("yaw", 0.0)
                pitch = att_result.get("pitch", 0.0)
                ear = fat_result.get("ear", 0.30)
                spine_angle = self._last_posture.get("spine_angle", 0.0)
                left_sh_y = self._last_posture.get("left_shoulder_y", 100.0)
                right_sh_y = self._last_posture.get("right_shoulder_y", 100.0)
                sh_width = self._last_posture.get("shoulder_width", 50.0)
                pose_available = self._last_posture.get("pose_available", False)
                phone_detected = self._last_phone.get("phone_found", False)
                phone_confidence = 0.85 if phone_detected else 0.0

                scores = self.score_engine.compute_all(
                    yaw=yaw, pitch=pitch, ear=ear,
                    spine_angle=spine_angle, left_sh_y=left_sh_y, right_sh_y=right_sh_y, sh_width=sh_width,
                    phone_detected=phone_detected, phone_confidence=phone_confidence,
                    pose_available=pose_available
                )
                self._last_ui_scores = scores

                # Correct L1 inputs with calibration baseline so TemporalEngine doesn't falsely trigger distraction
                att_result_corr = att_result.copy()
                if self.score_engine.calibrator.is_ready:
                    att_result_corr["yaw"] = yaw - self.score_engine.calibrator.yaw_baseline
                    att_result_corr["pitch"] = pitch - self.score_engine.calibrator.pitch_baseline

                # L2/L3/L4 processing (Smoothed L1 inputs)
                payload = self.engine.process(
                    att=self.att_smoother.smooth(att_result_corr),
                    fat=self.fat_smoother.smooth(fat_result),
                    pos=self.pos_smoother.smooth(self._last_posture),
                    phone=self._last_phone,
                )

                # Adaptive Sampling Strategy
                is_focused = payload.consolidated_states.work_mode == "focused"
                alert_active = payload.alert.should_alert
                state_stable_sec = payload.temporal_context.stable_state_for_sec if payload.temporal_context else 0.0
                
                is_stable = is_focused and not alert_active and state_stable_sec > 10.0
                
                # Requirements: 2-3s (unstable) vs 5-8s (stable)
                current_interval = 6.5 if is_stable else 2.5
                
                # Fast-path: if a discrete event occurred (mode change, phone, etc.), trigger immediately
                event_triggered = len(payload.events) > 0
                
                # Emit outputs on adaptive interval or immediate event
                if event_triggered or (now - self._last_emit) >= current_interval:
                    self._last_emit = now

                    clean_frame = self.formatter.format_clean_frame(payload, scores)
                    self.tracker.add_frame(clean_frame)

                    if self._frames_fp is not None:
                        self._frames_fp.write(json.dumps(clean_frame, ensure_ascii=False) + "\n")

                    # Backend snapshot (rich schema)
                    if self.send_to_backend:
                        snapshot = self.formatter.format_snapshot(payload, scores, clean_frame=clean_frame)
                        self.api_client.send_snapshot(snapshot)

                        # Discrete events only
                        for ev in payload.events or []:
                            if isinstance(ev, dict):
                                ev_type = ev.get("type")
                                if ev_type == "MODE_CHANGE":
                                    self.api_client.send_event(
                                        self.formatter.format_event(
                                            session_id=self.session_id,
                                            event_type="mode_change",
                                            level="info",
                                            description=ev.get("description", "Work mode changed"),
                                            metadata={"work_mode": payload.consolidated_states.work_mode},
                                        )
                                    )
                                elif ev_type == "ALERT" and payload.alert.should_alert:
                                    self.api_client.send_event(
                                        self.formatter.format_event(
                                            session_id=self.session_id,
                                            event_type="alert",
                                            level=payload.alert.level,
                                            description=ev.get("reason", "Alert"),
                                            metadata={"work_mode": payload.consolidated_states.work_mode},
                                        )
                                    )

                # Optional UI (minimal, every frame)
                if self.ui:
                    cs = payload.consolidated_states
                    phone_present = (cs.phone_state != "not_detected") or bool(self._last_phone.get("phone_found"))
                    distraction = (
                        cs.work_mode in {"brief_off_task", "phone_distraction", "social_distraction"}
                        or cs.attention_state == "distracted"
                    )

                    self.ui.draw(
                        frame,
                        state=cs.work_mode,
                        confidence=float(getattr(payload.reliability, "work_mode_confidence", 0.0) or 0.0),
                        phone_detected=phone_present,
                        factors={
                            "posture_state": cs.posture_state,
                            "fatigue_state": cs.fatigue_state,
                            "distraction": distraction,
                            "social_state": cs.social_state,
                            "eye_closed_instant": bool(fat_result.get("eye_closed")),
                            "yawn_instant": bool(fat_result.get("yawn_in_progress")),
                        },
                        scores=self._last_ui_scores,
                        reasoning=cs.reasoning_indices,
                    )
                    if self.ui.show(frame):
                        break

        finally:
            self.cap.release()
            if self.ui:
                self.ui.close()
            cv2.destroyAllWindows()

            summary = self.tracker.finalize()
            if self._summary_path is not None:
                try:
                    self._summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

            # Save a predictable copy of the final scores in the workspace root for easy user access
            try:
                workspace_summary_path = Path(__file__).resolve().parents[1] / "session_summary.json"
                workspace_summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[SmartFocus] Saved a copy of the session summary JSON to: {workspace_summary_path}")
            except Exception as e:
                print(f"[SmartFocus] Failed to save workspace copy: {e}")

            # Print session summary once at the end
            print("[SmartFocus] Session summary:")
            print(json.dumps(summary, ensure_ascii=False, indent=2))

            if self.send_to_backend:
                # Primary: Store in DB via dedicated stats endpoint
                self.api_client.finalize_session(self.session_id, summary)

                # Secondary: Record as event for history
                self.api_client.send_event(
                    self.formatter.format_event(
                        session_id=self.session_id,
                        event_type="session_summary",
                        level="info",
                        description="Session summary",
                        metadata=summary,
                    )
                )

            self._close_output_files()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--ui", action="store_false", dest="ui", default=True, help="Hide UI (default is shown)")
    parser.add_argument("--ui-debug", action="store_true", help="UI: also show numeric scores")

    # Legacy flags (kept for compatibility)
    parser.add_argument("--no-debug", action="store_true", help="(legacy) ignored; UI is off by default")
    parser.add_argument("--no-json", action="store_true", help="(legacy) do not send to backend")

    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Session duration in seconds (0=unlimited; press Q to quit when --ui is enabled)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--output-interval", type=float, default=0.5, help="Seconds between JSON outputs")

    args = parser.parse_args()

    pipeline = SmartFocusPipelineV3(
        ui=bool(args.ui),
        ui_debug=bool(args.ui_debug),
        send_to_backend=not bool(args.no_json),
        duration=int(args.duration),
        output_json=True,
        output_interval_sec=float(args.output_interval),
        camera_index=int(args.camera),
    )
    pipeline.run()
