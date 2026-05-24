"""
PhoneDetector - Level 1 Analyzer
Remplace le modèle TFLite MediaPipe par YOLOv26n (COCO class 67 = cell phone).
Plus robuste, meilleure précision, gestion des orientations multiples.
"""

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
PHONE_CLASS_ID       = 67      # COCO: "cell phone"
MIN_CONFIDENCE       = 0.35    # Increased to avoid detecting empty hands as phones
MIN_ASPECT_RATIO     = 0.7     # More permissive
MAX_ASPECT_RATIO     = 8.0     
MIN_REL_AREA         = 0.005   # Increased (approx 1500 pixels) to ensure phone is in the active workspace
MAX_REL_AREA         = 0.35    

class PhoneDetector:
    """
    Level 1 Analyzer: Détection instantanée de téléphone via YOLOv26n.
    Aucune persistance temporelle — géré par temporal_engine.
    """

    def __init__(self):
        self.model   = None
        self._ready  = False
        self._load_model()

    def _load_model(self):
        try:
            import torch
            
            # Patch torch.load to disable weights_only for ultralytics compatibility
            _original_torch_load = torch.load
            def patched_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return _original_torch_load(*args, **kwargs)
            torch.load = patched_load
            
            from ultralytics import YOLO
            
            # Load model
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model  = YOLO("yolo26n.pt")
            
            # Restore original torch.load
            torch.load = _original_torch_load
            
            self._ready = True
            print("[Phone L1] [OK] YOLOv26n chargé avec succès.")
        except ImportError:
            print("[Phone L1] [ERROR] ultralytics non installé. Lancez: pip install ultralytics")
        except Exception as e:
            print(f"[Phone L1] [ERROR] chargement modèle: {type(e).__name__}: {str(e)[:100]}")

    # ── Geometry filter ───────────────────────────────────────────────────────
    @staticmethod
    def _is_valid_phone_bbox(x1, y1, x2, y2, img_w, img_h) -> bool:
        """Filtre les détections non plausibles (trop petites, mauvais ratio)."""
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            return False

        # Ratio (on prend toujours long/court pour gérer portrait ET paysage)
        ratio = max(w, h) / (min(w, h) + 1e-6)
        if not (MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO):
            return False

        # Aire relative
        rel_area = (w * h) / (img_w * img_h + 1e-6)
        return MIN_REL_AREA <= rel_area <= MAX_REL_AREA

    # ── Main analyze ──────────────────────────────────────────────────────────
    def analyze(self, image, face_box=None) -> dict:
        """
        Retourne:
            phone_found  (bool)
            confidence   (float 0-1)
            bbox         (x1, y1, x2, y2) en pixels, ou None
            rel_area     (float) aire relative dans l'image
        """
        if not self._ready or self.model is None:
            return {"phone_found": False, "confidence": 0.0, "bbox": None, "rel_area": 0.0}

        img_h, img_w = image.shape[:2]

        try:
            results = self.model(
                image,
                classes=[PHONE_CLASS_ID],
                conf=MIN_CONFIDENCE,
                imgsz=416, 
                verbose=False,
                device='cpu' # FORCE CPU to avoid broken CUDA/Torchvision ops on Windows
            )
        except Exception as e:
            # Fallback if standard call fails due to ops errors
            try:
                # Try even lower resolution if it's a memory/op issue
                results = self.model(image, classes=[PHONE_CLASS_ID], conf=MIN_CONFIDENCE, imgsz=320, verbose=False, device='cpu')
            except Exception as e2:
                print(f"[Phone L1] Erreur inférence fatale: {e2}")
                return {"phone_found": False, "confidence": 0.0, "bbox": None, "rel_area": 0.0}

        best_conf = 0.0
        best_bbox = None
        best_area = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < MIN_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                if not self._is_valid_phone_bbox(x1, y1, x2, y2, img_w, img_h):
                    continue

                rel_area = ((x2 - x1) * (y2 - y1)) / (img_w * img_h + 1e-6)

                if conf > best_conf:
                    best_conf = conf
                    best_bbox = (x1, y1, x2, y2)
                    best_area = rel_area

        return {
            "phone_found": best_bbox is not None,
            "confidence":  round(best_conf, 3),
            "bbox":        best_bbox,
            "rel_area":    round(best_area, 4)
        }