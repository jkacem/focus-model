"""
Utilitaire pour lister toutes les caméras disponibles sur le système
"""
import cv2

def list_cameras(max_cameras=10):
    """Teste les index de caméra jusqu'à trouver les disponibles"""
    available = []
    
    for index in range(max_cameras):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"✓ Caméra {index}: {int(width)}x{int(height)} @ {int(fps)} fps")
            available.append(index)
            cap.release()
        else:
            # Optionally print unavailable
            pass
    
    return available

if __name__ == "__main__":
    print("=" * 50)
    print("Détection des caméras disponibles...")
    print("=" * 50)
    
    cameras = list_cameras(10)
    
    print("\n" + "=" * 50)
    if cameras:
        print(f"Caméras trouvées: {cameras}")
        print(f"\nUtilisez l'index avec:")
        print(f"  python main_cv.py --camera 0   # Caméra par défaut")
        if len(cameras) > 1:
            print(f"  python main_cv.py --camera {cameras[1]}   # {['Téléphone', 'Caméra USB', 'IP Webcam'][min(1, len(cameras)-2)]}")
    else:
        print("⚠ Aucune caméra détectée!")
    print("=" * 50)
