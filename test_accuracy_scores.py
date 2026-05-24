"""
Test de précision et de scores — Pipeline complet SmartFocus
============================================================
Mesure :
  • Précision détection téléphone  (FRONT / EAR / no phone)
  • Sorties ScoreEngine en temps réel
  • Efficacité : temps d'inférence par passe, FPS effectif

Usage :
    python test_accuracy_scores.py
    python test_accuracy_scores.py --duration 30   # session de 30s
Touches :
    Q     quitter + rapport final
    1     scénario "pas de téléphone" (ground truth = 0)
    2     scénario "téléphone devant" (ground truth = 1)
    3     scénario "téléphone à l'oreille" (ground truth = 1)
    S     screenshot
"""

import sys, time, argparse, collections
import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from analyzers.phone_detector   import PhoneDetector
from analyzers.attention_analyzer import AttentionAnalyzer
from analyzers.posture_analyzer  import PostureAnalyzer
from analyzers.fatigue_analyzer  import FatigueAnalyzer
from engine.smart_scoring        import ScoreEngine, ConcentrationScorer, PostureScorer, FatigueModulator

# ── Layout constants ───────────────────────────────────────────────────────────
BAR_W   = 220   # width of score bars panel
WIN_W   = 640 + BAR_W
WIN_H   = 480
FONT    = cv2.FONT_HERSHEY_SIMPLEX

SCENARIO_LABELS = {1: "NO PHONE", 2: "PHONE FRONT", 3: "PHONE EAR"}
SCENARIO_COLORS = {1: (0,200,0), 2: (0,0,255), 3: (0,140,255)}


# ── Drawing helpers ────────────────────────────────────────────────────────────

def draw_bar(panel, y, label, value, max_val=100.0, color=(80,200,80)):
    pct   = max(0.0, min(1.0, value / max_val))
    bar_w = int(pct * (BAR_W - 30))
    cv2.rectangle(panel, (10, y), (10 + bar_w, y + 16), color, -1)
    cv2.rectangle(panel, (10, y), (BAR_W - 20, y + 16), (100,100,100), 1)
    cv2.putText(panel, f"{label}: {value:.1f}", (10, y - 4), FONT, 0.38, (220,220,220), 1)


def draw_panel(scores: dict, phone_result: dict, stats: dict, scenario: int, fps: float) -> np.ndarray:
    panel = np.zeros((WIN_H, BAR_W, 3), dtype=np.uint8)
    panel[:] = (30, 30, 30)

    y = 12
    cv2.putText(panel, "── SCORES ──", (10, y), FONT, 0.45, (180,180,180), 1); y += 20

    conc  = scores.get("concentration", 0.0) or 0.0
    pos   = scores.get("posture",       0.0) or 0.0
    fat   = scores.get("fatigue",       0.0) or 0.0
    dist  = scores.get("distraction",   0.0) or 0.0
    glob  = scores.get("focus_global",  0.0) or 0.0

    draw_bar(panel, y, "Concentration", conc, color=(80,200,80));  y += 28
    draw_bar(panel, y, "Posture",       pos,  color=(80,160,220)); y += 28
    draw_bar(panel, y, "Fatigue",       fat,  color=(80,80,200));  y += 28
    draw_bar(panel, y, "Distraction",   dist, color=(60,60,200));  y += 28
    draw_bar(panel, y, "FOCUS GLOBAL",  glob, color=(0,220,180));  y += 34

    cv2.line(panel, (5, y), (BAR_W-5, y), (80,80,80), 1); y += 10

    # Phone detection info
    cv2.putText(panel, "── PHONE ──", (10, y), FONT, 0.45, (180,180,180), 1); y += 18
    found    = phone_result.get("phone_found", False)
    conf     = phone_result.get("confidence", 0.0)
    ear_mode = phone_result.get("ear_mode", False)
    mode_str = ("EAR" if ear_mode else "FRONT") if found else "---"
    color_ph = (0,140,255) if ear_mode else ((0,0,255) if found else (0,200,0))
    cv2.putText(panel, f"Found: {'YES' if found else 'NO'}  [{mode_str}]", (10, y), FONT, 0.40, color_ph, 1); y += 16
    cv2.putText(panel, f"Conf : {conf:.3f}", (10, y), FONT, 0.40, (180,180,180), 1); y += 20

    cv2.line(panel, (5, y), (BAR_W-5, y), (80,80,80), 1); y += 10

    # Accuracy stats
    cv2.putText(panel, "── ACCURACY ──", (10, y), FONT, 0.45, (180,180,180), 1); y += 18
    for sc, lbl in SCENARIO_LABELS.items():
        s = stats[sc]
        total = s["total"]
        if total == 0:
            acc_str = "n/a"
        else:
            correct = s["tp"] + s["tn"]
            acc_str = f"{correct/total*100:.0f}%  ({correct}/{total})"
        cv2.putText(panel, f"{lbl[:10]}: {acc_str}", (10, y), FONT, 0.35, SCENARIO_COLORS[sc], 1); y += 16

    # False positives / negatives
    s1 = stats[1]
    fp_rate = (s1["fp"] / s1["total"] * 100) if s1["total"] > 0 else 0.0
    cv2.putText(panel, f"FP rate: {fp_rate:.0f}%", (10, y), FONT, 0.38, (200,100,100), 1); y += 16

    fn_total = stats[2]["fn"] + stats[3]["fn"]
    fn_denom = stats[2]["total"] + stats[3]["total"]
    fn_rate  = (fn_total / fn_denom * 100) if fn_denom > 0 else 0.0
    cv2.putText(panel, f"FN rate: {fn_rate:.0f}%", (10, y), FONT, 0.38, (100,150,200), 1); y += 20

    cv2.line(panel, (5, y), (BAR_W-5, y), (80,80,80), 1); y += 10

    # Efficiency
    cv2.putText(panel, "── EFFICIENCY ──", (10, y), FONT, 0.45, (180,180,180), 1); y += 18
    cv2.putText(panel, f"FPS        : {fps:.1f}", (10, y), FONT, 0.38, (180,180,180), 1); y += 14
    cv2.putText(panel, f"Inf pass1  : {stats['inf_p1_ms']:.0f} ms", (10, y), FONT, 0.38, (180,180,180), 1); y += 14
    cv2.putText(panel, f"Inf pass2  : {stats['inf_p2_ms']:.0f} ms", (10, y), FONT, 0.38, (180,180,180), 1); y += 14
    cv2.putText(panel, f"Total det  : {stats['total_det']}", (10, y), FONT, 0.38, (180,180,180), 1); y += 14

    cv2.line(panel, (5, y), (BAR_W-5, y), (80,80,80), 1); y += 10

    # Scenario indicator
    sc_lbl   = SCENARIO_LABELS.get(scenario, "---")
    sc_color = SCENARIO_COLORS.get(scenario, (180,180,180))
    cv2.putText(panel, f"Scenario: {sc_lbl}", (10, y), FONT, 0.42, sc_color, 1); y += 16
    cv2.putText(panel, "1=no phone 2=front 3=ear", (10, y), FONT, 0.32, (120,120,120), 1)

    return panel


def draw_overlay(frame, phone_result, face_box):
    out = frame.copy()
    # Face box
    if face_box is not None:
        fx1,fy1,fx2,fy2 = face_box
        cv2.rectangle(out,(fx1,fy1),(fx2,fy2),(200,150,0),1)
    # Phone box
    bbox = phone_result.get("bbox")
    if bbox:
        x1,y1,x2,y2 = bbox
        ear = phone_result.get("ear_mode",False)
        color = (0,140,255) if ear else (0,0,255)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        label = f"{'EAR ' if ear else ''}PHONE {phone_result['confidence']:.2f}"
        cv2.putText(out,label,(x1,max(y1-8,16)),FONT,0.55,color,2)
    return out


# ── Stats helpers ──────────────────────────────────────────────────────────────

def fresh_stats():
    return {sc: {"total":0,"tp":0,"tn":0,"fp":0,"fn":0}
            for sc in SCENARIO_LABELS} | {
        "inf_p1_ms": 0.0, "inf_p2_ms": 0.0, "total_det": 0
    }

def update_stats(stats, scenario, phone_found, inf_ms_p1, inf_ms_p2):
    s = stats[scenario]
    s["total"] += 1
    gt_phone = (scenario in (2, 3))   # ground truth: phone present?
    if gt_phone and phone_found:
        s["tp"] += 1
    elif not gt_phone and not phone_found:
        s["tn"] += 1
    elif not gt_phone and phone_found:
        s["fp"] += 1
    elif gt_phone and not phone_found:
        s["fn"] += 1
    if phone_found:
        stats["total_det"] += 1
    # Running average of inference times
    alpha = 0.1
    stats["inf_p1_ms"] = (1-alpha)*stats["inf_p1_ms"] + alpha*inf_ms_p1
    stats["inf_p2_ms"] = (1-alpha)*stats["inf_p2_ms"] + alpha*inf_ms_p2


def print_final_report(stats, duration):
    print("\n" + "="*55)
    print("          RAPPORT FINAL — SmartFocus Accuracy Test")
    print("="*55)
    print(f"  Durée session : {duration:.1f}s")
    for sc, lbl in SCENARIO_LABELS.items():
        s = stats[sc]
        total = s["total"]
        if total == 0:
            print(f"  {lbl:15s} : pas de données")
            continue
        acc = (s["tp"]+s["tn"]) / total * 100
        pr_str = f"TP={s['tp']} TN={s['tn']} FP={s['fp']} FN={s['fn']}"
        print(f"  {lbl:15s} : Acc={acc:.0f}%  {pr_str}  (n={total})")

    s1 = stats[1]
    fp_rate = (s1["fp"]/s1["total"]*100) if s1["total"]>0 else 0.0
    fn_d = stats[2]["fn"]+stats[3]["fn"]
    fn_n = stats[2]["total"]+stats[3]["total"]
    fn_rate = (fn_d/fn_n*100) if fn_n>0 else 0.0
    print(f"\n  Taux faux positifs : {fp_rate:.1f}%")
    print(f"  Taux faux négatifs : {fn_rate:.1f}%")
    print(f"\n  Inférence passe 1 (globale) : {stats['inf_p1_ms']:.0f} ms")
    print(f"  Inférence passe 2 (crop)    : {stats['inf_p2_ms']:.0f} ms")
    print(f"  Détections totales          : {stats['total_det']}")
    print("="*55)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=0)
    parser.add_argument("--camera",   type=int, default=0)
    args = parser.parse_args()

    print("=== SmartFocus — Accuracy & Score Test ===")
    print("Chargement des modèles (patience)...")

    attention = AttentionAnalyzer()
    fatigue   = FatigueAnalyzer()
    posture   = PostureAnalyzer()
    detector  = PhoneDetector()

    conc_sc = ConcentrationScorer()
    pos_sc  = PostureScorer()
    fat_mod = FatigueModulator()
    score_eng = ScoreEngine(conc_sc, pos_sc, fat_mod)
    fat_mod.start_session()

    print("Ouverture caméra...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Prêt. Touches : 1=no phone  2=front  3=ear  Q=quitter  S=screenshot")
    print()

    stats    = fresh_stats()
    scenario = 1    # par défaut : pas de téléphone
    scores   = {"concentration":100,"posture":100,"fatigue":0,"distraction":0,"focus_global":100}
    last_phone  = {"phone_found":False,"confidence":0.0,"bbox":None,"rel_area":0.0,"ear_mode":False}
    last_att    = {}
    last_fat    = {}
    last_pos    = {}
    face_box_full = None

    frame_n = 0
    ATT_SKIP    = 4
    FAT_SKIP    = 4
    POS_SKIP    = 10
    PHONE_SKIP  = 6

    t0, fps, cnt = time.time(), 0.0, 0
    t_start = time.time()

    # Instrument infer_ear_crop to measure pass-2 time separately
    _orig_crop = PhoneDetector._infer_ear_crop
    p2_times = []
    def _timed_crop(self, image, face_box, img_w, img_h):
        t = time.time()
        r = _orig_crop(self, image, face_box, img_w, img_h)
        p2_times.append((time.time()-t)*1000)
        return r
    PhoneDetector._infer_ear_crop = _timed_crop

    CAL_SEC = 4
    print(f"Calibration {CAL_SEC}s — regardez la caméra...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        cnt     += 1
        now      = time.time()
        elapsed  = now - t_start
        calibrating = elapsed < CAL_SEC

        if now - t0 >= 1.0:
            fps = cnt / (now - t0)
            cnt = 0
            t0  = now

        if args.duration > 0 and not calibrating and (elapsed - CAL_SEC) >= args.duration:
            print(f"\n[Auto-stop] {args.duration}s écoulées.")
            break

        small = cv2.resize(frame, (160, 120))

        if frame_n % ATT_SKIP == 0:
            last_att = attention.analyze(small, calibrating=calibrating)
        if frame_n % FAT_SKIP == 0:
            yaw_deg = last_att.get("yaw", 0.0) if not calibrating else 0.0
            last_fat = fatigue.analyze(small, calibrating=calibrating, yaw_deg=yaw_deg)
        if frame_n % POS_SKIP == 0:
            last_pos = posture.analyze(small, calibrating=calibrating)

        # face_box rescaled to full frame
        raw_fb = last_att.get("face_bbox")
        if raw_fb is not None:
            sx, sy = frame.shape[1]/160, frame.shape[0]/120
            fx1,fy1,fx2,fy2 = raw_fb
            face_box_full = (int(fx1*sx),int(fy1*sy),int(fx2*sx),int(fy2*sy))
        else:
            face_box_full = None

        if not calibrating and frame_n % PHONE_SKIP == 0:
            p2_times.clear()
            t_p1 = time.time()
            last_phone = detector.analyze(frame, face_box=face_box_full)
            inf_total  = (time.time()-t_p1)*1000
            inf_p2     = p2_times[0] if p2_times else 0.0
            inf_p1     = inf_total - inf_p2
            update_stats(stats, scenario, last_phone["phone_found"], inf_p1, inf_p2)

            ph = last_phone
            tag = ("[EAR]" if ph["ear_mode"] else "[FRONT]") if ph["phone_found"] else ""
            print(f"  sc={scenario} {'DET '+tag+' c='+str(round(ph['confidence'],2)) if ph['phone_found'] else 'no det':30s}"
                  f"  p1={inf_p1:.0f}ms p2={inf_p2:.0f}ms", end="\r")

        # Scores (every frame after calibration)
        if not calibrating:
            yaw   = last_att.get("yaw",  0.0)
            pitch = last_att.get("pitch", 0.0)
            ear   = last_fat.get("ear",  0.30)
            spine = last_pos.get("spine_angle", 0.0)
            lsh   = last_pos.get("left_shoulder_y",  100.0)
            rsh   = last_pos.get("right_shoulder_y", 100.0)
            shw   = last_pos.get("shoulder_width",    50.0)
            pose_ok = last_pos.get("pose_available", False)
            ph_det  = last_phone.get("phone_found", False)
            ph_conf = 0.85 if ph_det else 0.0

            scores = score_eng.compute_all(
                yaw=yaw, pitch=pitch, ear=ear,
                spine_angle=spine, left_sh_y=lsh, right_sh_y=rsh, sh_width=shw,
                phone_detected=ph_det, phone_confidence=ph_conf,
                pose_available=pose_ok
            )

        # Build display
        cam_draw = draw_overlay(frame, last_phone, face_box_full)

        if calibrating:
            prog = min(1.0, elapsed / CAL_SEC)
            cv2.rectangle(cam_draw, (0,0),(frame.shape[1],4),(80,80,80),-1)
            cv2.rectangle(cam_draw, (0,0),(int(frame.shape[1]*prog),4),(0,220,180),-1)
            cv2.putText(cam_draw, f"Calibration... {int(prog*100)}%",
                        (10,30), FONT, 0.7, (0,220,180), 2)

        panel = draw_panel(scores, last_phone, stats, scenario, fps)
        canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
        canvas[:, :640] = cam_draw
        canvas[:, 640:] = panel

        cv2.imshow("SmartFocus — Accuracy & Score Test", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("1"):
            scenario = 1
            print(f"\n[Scenario] NO PHONE")
        elif key == ord("2"):
            scenario = 2
            print(f"\n[Scenario] PHONE FRONT")
        elif key == ord("3"):
            scenario = 3
            print(f"\n[Scenario] PHONE EAR")
        elif key == ord("s"):
            fname = f"accuracy_cap_{int(time.time())}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"\n[Saved] {fname}")

    cap.release()
    cv2.destroyAllWindows()

    PhoneDetector._infer_ear_crop = _orig_crop  # restore

    duration = time.time() - t_start
    print_final_report(stats, duration)


if __name__ == "__main__":
    main()
