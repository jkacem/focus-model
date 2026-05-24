"""Benchmark scientifique du ScoreEngine v4."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from engine.smart_scoring import ScoreEngine

eng = ScoreEngine()
eng.start_session()
for _ in range(80):
    eng.calibrator.feed(0.0, 0.0, 0.28, 0.0, 100.0, 100.0, 50.0)

scenarios = [
    ("Concentre (ideal)",      dict(yaw=1,  pitch=-2,  ear=0.31, spine_angle=1,  left_sh_y=100, right_sh_y=100, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=True)),
    ("Lecture (tete baissee)", dict(yaw=2,  pitch=-20, ear=0.27, spine_angle=8,  left_sh_y=100, right_sh_y=101, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=True)),
    ("Distrait (regard cote)", dict(yaw=42, pitch=2,   ear=0.29, spine_angle=2,  left_sh_y=100, right_sh_y=100, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=True)),
    ("Tel a oreille",          dict(yaw=8,  pitch=0,   ear=0.28, spine_angle=3,  left_sh_y=100, right_sh_y=100, sh_width=50, phone_detected=True,  phone_confidence=0.78, pose_available=True)),
    ("Mauvaise posture",       dict(yaw=3,  pitch=-5,  ear=0.30, spine_angle=28, left_sh_y=100, right_sh_y=120, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=True)),
    ("Fatigue (yeux mi-ferm)", dict(yaw=2,  pitch=-3,  ear=0.16, spine_angle=4,  left_sh_y=100, right_sh_y=100, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=True)),
    ("Somnolent (EAR=0.12)",   dict(yaw=5,  pitch=2,   ear=0.12, spine_angle=10, left_sh_y=100, right_sh_y=105, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=True)),
    ("Sans pose",              dict(yaw=3,  pitch=-4,  ear=0.29, spine_angle=0,  left_sh_y=100, right_sh_y=100, sh_width=50, phone_detected=False, phone_confidence=0.0,  pose_available=False)),
]

print()
print(f"{'Scenario':<26} {'Attn':>5} {'Vig':>5} {'Post':>5} {'Distr':>6} {'FOCUS':>6}")
print("-" * 60)
for name, kw in scenarios:
    r = eng.compute_all(**kw)
    post = f"{r['posture']:.1f}" if r["posture"] is not None else "N/A"
    print(f"{name:<26} {r['concentration']:>5.1f} {r['vigilance']:>5.1f} {post:>5} {r['distraction']:>6.1f} {r['focus_global']:>6.1f}")

print()
print("Validation logique :")
r_good = [r for name, kw in scenarios[:2] for _ in [None] if (r := eng.compute_all(**kw))]
r_bad  = [r for name, kw in scenarios[2:6] for _ in [None] if (r := eng.compute_all(**kw))]
focus_good = sum(r["focus_global"] for r in r_good) / len(r_good)
focus_bad  = sum(r["focus_global"] for r in r_bad)  / len(r_bad)
print(f"  Focus moyen etats positifs : {focus_good:.1f}")
print(f"  Focus moyen etats negatifs : {focus_bad:.1f}")
assert focus_good > focus_bad, "ERREUR : les etats positifs devraient avoir un meilleur focus"
print("  OK — etats positifs > etats negatifs")
