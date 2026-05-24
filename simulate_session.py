import sys
import os
from pathlib import Path
from unittest.mock import patch
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

class MockTime:
    def __init__(self):
        self.current_time = time.time()
    def __call__(self):
        return self.current_time
    def tick(self, dt):
        self.current_time += dt

mock_time = MockTime()

with patch('time.time', side_effect=mock_time):
    from engine.smart_scoring import ScoreEngine, ConcentrationScorer, PostureScorer, FatigueModulator

    def print_state(t, name, scores):
        mins = int(t // 60)
        secs = int(t % 60)
        print(f"[{mins:02d}:{secs:02d}] {name:<28} | Focus: {scores['focus_global']:>5.1f} | Conc: {scores['concentration']:>5.1f} | Post: {scores['posture']:>5.1f} | Fatg: {scores['fatigue']:>5.1f} | Dist: {scores['distraction']:>5.1f}")

    def run_session():
        concentration = ConcentrationScorer()
        posture = PostureScorer()
        fatigue = FatigueModulator()
        engine = ScoreEngine(concentration, posture, fatigue)
        
        fps = 15
        dt = 1.0 / fps
        total_time = 0.0
        
        fatigue.start_session()
        
        print("=== DÉBUT DE LA SESSION DE SIMULATION (60 Minutes) ===")
        print(f"{'Temps':<7} {'État Actuel':<28} | {'Focus':>5} | {'Conc':>5} | {'Post':>5} | {'Fatg':>5} | {'Dist':>5}")
        print("-" * 88)
        
        def simulate_period(minutes, name, yaw, pitch, ear, spine_angle, phone_detected, phone_confidence=0.0):
            nonlocal total_time
            frames = int(minutes * 60 * fps)
            for i in range(frames):
                mock_time.tick(dt)
                total_time += dt
                scores = engine.compute_all(
                    yaw=yaw, pitch=pitch, ear=ear,
                    spine_angle=spine_angle, left_sh_y=100.0, right_sh_y=100.0, sh_width=50.0,
                    phone_detected=phone_detected, phone_confidence=phone_confidence
                )
                
                # Afficher le statut chaque minute
                if (i + 1) % (60 * fps) == 0:
                    print_state(total_time, name, scores)

        # 1. Calibration & Début parfait (5 min)
        simulate_period(5.0, "Focalisé (Parfait)", yaw=0.0, pitch=0.0, ear=0.32, spine_angle=0.0, phone_detected=False)
        
        # 2. Lecture (15 min) - Tête un peu baissée
        simulate_period(15.0, "Lecture", yaw=0.0, pitch=15.0, ear=0.28, spine_angle=10.0, phone_detected=False)
        
        # 3. Distraction Téléphone (5 min)
        simulate_period(5.0, "Téléphone", yaw=20.0, pitch=25.0, ear=0.25, spine_angle=15.0, phone_detected=True, phone_confidence=0.85)
        
        # 4. Retour au travail mais fatigue s'installe (20 min)
        simulate_period(20.0, "Travail (Fatigue modérée)", yaw=5.0, pitch=5.0, ear=0.21, spine_angle=5.0, phone_detected=False)
        
        # 5. Très fatigué et mauvaise posture (15 min)
        simulate_period(15.0, "Somnolence & Mauvaise Posture", yaw=5.0, pitch=10.0, ear=0.18, spine_angle=35.0, phone_detected=False)
        
        print("-" * 88)
        print("=== FIN DE LA SESSION ===")

if __name__ == "__main__":
    run_session()
