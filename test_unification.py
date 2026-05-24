import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.session_tracker import SessionTracker

def test_unification():
    print("Running Unification Test...")
    tracker = SessionTracker(session_id="test")

    # 50 frames à GlobalFocus=20 (distrait)
    for _ in range(50):
        # We simulate the format expected by add_frame
        frame = {
            "scores": {
                "global_focus": 20.0,
                "attention_score": 20.0,
                "posture_score": 100.0,
                "fatigue_modulator": 0.8
            }
        }
        tracker.add_frame(frame)

    # 50 frames à GlobalFocus=75 (focalisé)
    for _ in range(50):
        frame = {
            "scores": {
                "global_focus": 75.0,
                "attention_score": 75.0,
                "posture_score": 100.0,
                "fatigue_modulator": 0.8
            }
        }
        tracker.add_frame(frame)

    metrics = tracker.finalize()

    print(f"Focus time ratio: {metrics['focus_time_ratio']}")
    print(f"Distraction time ratio: {metrics['distraction_time_ratio']}")

    # focus_time_ratio doit refléter les 50 frames >= 60
    assert abs(metrics["focus_time_ratio"] - 0.50) < 0.05, f"Focus ratio mismatch: {metrics['focus_time_ratio']}"

    # distraction_time_ratio doit refléter les 50 frames < 40
    assert abs(metrics["distraction_time_ratio"] - 0.50) < 0.05, f"Distraction ratio mismatch: {metrics['distraction_time_ratio']}"

    # Les deux ne somment pas forcément à 1.0 — c'est voulu (même si ici c'est 0.5 et 0.5)
    print("OK — focus_ratio et distraction_ratio cohérents avec GlobalFocus")

if __name__ == "__main__":
    test_unification()
