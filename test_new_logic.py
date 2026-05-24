import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.smart_scoring import ConcentrationScorer, PostureScorer, FatigueModulator, ScoreEngine

def test_assertions():
    engine = ScoreEngine(ConcentrationScorer(), PostureScorer(), FatigueModulator())
    
    print("Testing Calibration")
    # Feed 5 seconds (15 fps = 75 frames) of "normal" behavior
    for _ in range(75):
        scores = engine.compute_all(yaw=15.0, pitch=-10.0, ear=0.25, spine_angle=12.0, left_sh_y=50.0, right_sh_y=52.0, sh_width=100.0, phone_detected=False)
    
    assert engine.calibrator.is_ready, "Calibration did not complete!"
    print(f"Calibration termin\u00e9e \u2014 yaw_baseline={engine.calibrator.yaw_baseline:.2f} pitch_baseline={engine.calibrator.pitch_baseline:.2f} ear_baseline={engine.calibrator.ear_baseline:.2f}")

    print("Testing Rule: conditions idéales post-calibration -> focus elevé")
    # Simulate perfect behavior relative to the baseline
    focus_global_conditions_ideales = 0.0
    for _ in range(300):
        scores = engine.compute_all(
            yaw=engine.calibrator.yaw_baseline,
            pitch=engine.calibrator.pitch_baseline,
            ear=engine.calibrator.ear_baseline,
            spine_angle=engine.calibrator.spine_baseline,
            left_sh_y=50.0, right_sh_y=52.0, sh_width=100.0,
            phone_detected=False
        )
        focus_global_conditions_ideales = scores["focus_global"]
        concentration = scores["concentration"]
        
    assert scores["concentration"] >= 70.0, f"Concentration trop basse pour conditions id\u00e9ales: {scores['concentration']}"
    assert scores["distraction"] <= 20.0, f"Distraction trop haute pour conditions id\u00e9ales: {scores['distraction']}"
    assert scores["focus_global"] >= 55.0, f"Focus trop bas pour conditions id\u00e9ales: {scores['focus_global']}"
    
    print(f"PASS (focus={focus_global_conditions_ideales}, conc={scores['concentration']}, dist={scores['distraction']})")

    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_assertions()
