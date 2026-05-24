import numpy as np

"""
DETECTION D'ACTIVITE VOCALE (FR):
Ce module simule ou détecte l'activité vocale (le fait de parler).
Il permet de distinguer si l'utilisateur est en train de s'expliquer un cours (auto-explication).

VOICE ACTIVITY DETECTION (EN):
This module detects speech activity. 
It helps distinguish if the user is self-explaining course material.
"""

class AudioDetector:
    def __init__(self):
        self.is_speaking = False
        # Note: Pour un PFE, une simulation basée sur l'énergie sonore simple suffit
        # ou un flag mock pour la démonstration.
        
    def detect(self):
        """
        Retourne True si une activité vocale est détectée.
        Pour le moment, retourne False (Mock).
        """
        return self.is_speaking

    def set_mock_speech(self, status: bool):
        """Permet de simuler la parole pendant les tests."""
        self.is_speaking = status
