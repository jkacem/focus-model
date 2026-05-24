import random
import time

class SimulatedCamera:
    def get_frame_analysis(self):
        """
        Simulates AI analysis of a camera frame.
        Returns a dictionary with posture and fatigue scores.
        """
        # Simulate some logic: usually good, occasionally bad
        posture_score = random.uniform(0.5, 1.0)
        fatigue_score = random.uniform(0.6, 1.0)
        
        # Randomly trigger a low score for testing
        if random.random() < 0.1:
            posture_score = random.uniform(0.2, 0.4)
        if random.random() < 0.05:
            fatigue_score = random.uniform(0.1, 0.3)
            
        return {
            "posture": round(posture_score, 2),
            "fatigue": round(fatigue_score, 2)
        }

class SimulatedMicrophone:
    def get_audio_analysis(self):
        """
        Simulates audio distraction detection.
        """
        distraction_score = random.uniform(0.7, 1.0)
        
        if random.random() < 0.05:
            distraction_score = random.uniform(0.2, 0.5)
            
        return {
            "distraction": round(distraction_score, 2)
        }
