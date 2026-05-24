"""
Test YOLO26n PhoneDetector avec face_box réel (AttentionAnalyzer).
Simule exactement le pipeline main_cv.py.
Appuie sur Q pour quitter, S pour screenshot.
"""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from analyzers.phone_detector import PhoneDetector
from analyzers.attention_analyzer import AttentionAnalyzer


def draw_result(frame: np.ndarray, result: dict, face_box, fps: float) -> np.ndarray:
    h, w = frame.shape[:2]
    out = frame.copy()

    phone_found = result.get("phone_found", False)
    conf        = result.get("confidence", 0.0)
    bbox        = result.get("bbox")
    rel_area    = result.get("rel_area", 0.0)
    ear_mode    = result.get("ear_mode", False)

    # Face box (bleu)
    if face_box is not None:
        fx1, fy1, fx2, fy2 = face_box
        cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (200, 150, 0), 1)

        # Zones oreille (tirets cyan)
        face_ww = fx2 - fx1
        face_hh = fy2 - fy1
        lat = int(face_ww * 1.5)
        ear_top = max(0, int(fy1 - face_hh * 0.25))
        ear_bot = min(h, int(fy2 + int(face_hh * 0.10)))
        # gauche
        cv2.rectangle(out, (max(0, fx1 - lat), ear_top),
                      (min(w, fx1 + int(face_ww * 0.15)), ear_bot), (0, 200, 200), 1)
        # droite
        cv2.rectangle(out, (max(0, fx2 - int(face_ww * 0.15)), ear_top),
                      (min(w, fx2 + lat), ear_bot), (0, 200, 200), 1)

    # Phone bounding box
    if bbox:
        x1, y1, x2, y2 = bbox
        color = (0, 140, 255) if ear_mode else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{'EAR ' if ear_mode else ''}PHONE {conf:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Status bar
    if phone_found:
        mode_tag     = " [EAR]" if ear_mode else " [FRONT]"
        status_color = (0, 140, 255) if ear_mode else (0, 0, 255)
        status_text  = f"PHONE{mode_tag}  conf={conf:.2f}  area={rel_area:.4f}"
    else:
        status_color = (0, 200, 0)
        status_text  = "no phone"

    cv2.rectangle(out, (0, 0), (w, 32), (20, 20, 20), -1)
    cv2.putText(out, status_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

    # Legend
    cv2.putText(out, "cyan=ear zone  blue=face  red=front  orange=ear",
                (10, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
    cv2.putText(out, f"YOLO26n  {fps:.1f} fps", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return out


def main():
    print("=== Test YOLO26n PhoneDetector (avec face_box) ===")
    print("Chargement des modèles...")

    attention = AttentionAnalyzer()
    detector  = PhoneDetector()

    if not detector._ready:
        print("[ERREUR] PhoneDetector non prêt.")
        return

    print("Ouverture caméra...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d'ouvrir la caméra.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("Prêt. Appuie sur Q pour quitter, S pour sauvegarder.")
    print("Amène le téléphone de face, de côté, ou à l'oreille.\n")

    last_phone  = {"phone_found": False, "confidence": 0.0,
                   "bbox": None, "rel_area": 0.0, "ear_mode": False}
    last_att    = {}
    face_box_full = None

    frame_n   = 0
    ATT_SKIP  = 4    # attention toutes les 4 frames (pour obtenir face_box)
    PHONE_SKIP = 6   # phone toutes les 6 frames

    t0, fps = time.time(), 0.0
    cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_n += 1
        cnt += 1
        now = time.time()
        if now - t0 >= 1.0:
            fps = cnt / (now - t0)
            cnt = 0
            t0 = now

        # Petite image pour attention (léger)
        small = cv2.resize(frame, (160, 120))

        if frame_n % ATT_SKIP == 0:
            last_att = attention.analyze(small, calibrating=False)

        # Convertir face_box de l'espace 160×120 vers 640×480
        raw_fb = last_att.get("face_bbox")
        if raw_fb is not None:
            sx = frame.shape[1] / 160
            sy = frame.shape[0] / 120
            fx1, fy1, fx2, fy2 = raw_fb
            face_box_full = (int(fx1*sx), int(fy1*sy), int(fx2*sx), int(fy2*sy))
        else:
            face_box_full = None

        if frame_n % PHONE_SKIP == 0:
            t_inf = time.time()
            last_phone = detector.analyze(frame, face_box=face_box_full)
            inf_ms = (time.time() - t_inf) * 1000

            phone_found = last_phone.get("phone_found", False)
            conf        = last_phone.get("confidence", 0.0)
            ear_mode    = last_phone.get("ear_mode", False)
            if phone_found:
                tag = "[EAR]" if ear_mode else "[FRONT]"
                print(f"  [DETECTED {tag} conf={conf:.2f}]  {inf_ms:.0f}ms", end="\r")
            else:
                print(f"  [not detected]  {inf_ms:.0f}ms", end="\r")

        display = draw_result(frame, last_phone, face_box_full, fps)
        cv2.imshow("YOLO26n Phone Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"yolo26_capture_{int(time.time())}.jpg"
            cv2.imwrite(fname, display)
            print(f"\n[Saved] {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== Test terminé ===")


if __name__ == "__main__":
    main()
