import sys
import os
from pathlib import Path
import time

# Add the current directory to sys.path to allow importing local modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine.smart_scoring import ScoreEngine, ConcentrationScorer, PostureScorer, FatigueModulator

def print_test_result(name, scores):
    print(f"\n--- TEST: {name} ---")
    print(f"  RESULT SCORES:")
    print(f"    Concentration Score: {scores['concentration']}")
    print(f"    Posture Score:       {scores['posture']}")
    print(f"    Fatigue Score:       {scores['fatigue']}")
    print(f"    Distraction Score:   {scores['distraction']}")
    print(f"    GLOBAL FOCUS:        {scores['focus_global']} / 100")

def run_tests():
    concentration = ConcentrationScorer()
    posture = PostureScorer()
    fatigue = FatigueModulator()
    engine = ScoreEngine(concentration, posture, fatigue)
    
    # Simulate a session
    fatigue.start_session()
    
    # Calibration is required first (75 frames at 15fps = 5 seconds)
    print("Running calibration (75 frames)...")
    for _ in range(80): 
        engine.compute_all(
            yaw=0.0, pitch=0.0, ear=0.32,
            spine_angle=0.0, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
            phone_detected=False
        )

    # Helper function to run a scenario for several frames to let EMA settle
    def run_scenario(name, yaw, pitch, ear, spine_angle, left_sh_y, right_sh_y, sh_width, phone_detected, phone_confidence=0.0, frames=45):
        result = None
        for _ in range(frames):
            # simulate passage of time for posture scorer
            time.sleep(0.01)
            result = engine.compute_all(
                yaw=yaw, pitch=pitch, ear=ear,
                spine_angle=spine_angle, left_sh_y=left_sh_y, right_sh_y=right_sh_y, sh_width=sh_width,
                phone_detected=phone_detected, phone_confidence=phone_confidence
            )
        print_test_result(name, result)

    # 1. PARFAIT (FOCUSED)
    run_scenario("FOCALISÉ (PARFAIT)",
        yaw=0.0, pitch=0.0, ear=0.32,
        spine_angle=0.0, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
        phone_detected=False
    )

    # 2. LECTURE (READING)
    run_scenario("LECTURE (Tête un peu baissée)",
        yaw=0.0, pitch=15.0, ear=0.28,
        spine_angle=10.0, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
        phone_detected=False
    )

    # 3. DISTRAIT (TÉLÉPHONE)
    run_scenario("DISTRACTION TÉLÉPHONE",
        yaw=20.0, pitch=25.0, ear=0.25,
        spine_angle=15.0, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
        phone_detected=True, phone_confidence=0.85
    )

    # 4. FATIGUÉ (HAUT)
    run_scenario("FATIGUE ÉLEVÉE (Yeux presque fermés)",
        yaw=5.0, pitch=5.0, ear=0.19,
        spine_angle=20.0, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
        phone_detected=False, frames=90 # Needs longer for PERCLOS
    )

    # 5. MAUVAISE POSTURE
    run_scenario("MAUVAISE POSTURE (Dos courbé et asymétrique)",
        yaw=0.0, pitch=0.0, ear=0.30,
        spine_angle=45.0, left_sh_y=120.0, right_sh_y=90.0, sh_width=50.0,
        phone_detected=False
    )

if __name__ == "__main__":
    run_tests()
