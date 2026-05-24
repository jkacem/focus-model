import sys
import os
from pathlib import Path

# Add the current directory to sys.path to allow importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.smart_scoring import ScoreEngine, AttentionScorer, VigilanceScorer, PostureScorer, FatigueModulator
from engine.session_tracker import SessionTracker

def test_bug_fixes():
    # Setup scoring engine
    attention = AttentionScorer()
    vigilance = VigilanceScorer()
    posture = PostureScorer()
    fatigue = FatigueModulator()
    engine = ScoreEngine(attention, vigilance, posture, fatigue)
    fatigue.start_session()
    
    tracker = SessionTracker(session_id="test")

    # Simulate 10 frames of 100% focus (GlobalFocus >= 60)
    for _ in range(10):
        scores = engine.compute_all(
            yaw=0.0, pitch=0.0, ear=0.30,
            spine_angle=0.0, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
            blink_rate=15.0, phone_detected=False
        )
        # Note: formatting mock for tracker input
        clean_frame = {"scores": scores}
        tracker.add_frame(clean_frame)
        
    summary = tracker.finalize()
    
    # Bug 2 Test: focus_time_ratio should be 1.0 (10/10 frames focused)
    assert summary["focus_time_ratio"] == 1.0, f"Bug 2 Failed: focus_time_ratio is {summary['focus_time_ratio']}"
    print("[OK] Bug 2: focus_time_ratio est bien calculé depuis GlobalFocus >= 60")
    
    # Bug 3 Test: fatigue_modulator should be in breakdown, not fatigue_score
    assert "fatigue_modulator" in summary["breakdown"], "Bug 3 Failed: fatigue_modulator missing"
    assert "fatigue_score" not in summary["breakdown"], "Bug 3 Failed: fatigue_score still present"
    print(f"[OK] Bug 3: Breakdown utilise fatigue_modulator ({summary['breakdown']['fatigue_modulator']})")
    
    # Bug 4 Test: final_score must be bounded
    assert 0.0 <= summary["final_score"] <= 100.0, "Bug 4 Failed: final_score out of bounds"
    print(f"[OK] Bug 4: final_score correctement borné ({summary['final_score']})")

if __name__ == "__main__":
    test_bug_fixes()
