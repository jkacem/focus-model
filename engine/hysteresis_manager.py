import time
from typing import Dict, Any

class HysteresisState:
    """
    Handles asymmetric state transitions using a cumulative confidence (leaky bucket) approach.
    This prevents a single positive/negative frame from resetting the entire transition timer.
    """
    
    def __init__(self, initial_state: str, enter_sec: float, exit_sec: float):
        self.current_state = initial_state
        self.default_state = initial_state
        self.enter_sec = enter_sec
        self.exit_sec = exit_sec
        
        self._candidate_state: str = initial_state
        self._confidence: float = 0.0
        self._last_time: float = 0.0

    def update(self, now: float, observed_state: str) -> str:
        dt = now - self._last_time if self._last_time > 0 else 0.0
        self._last_time = now
        dt = min(dt, 0.5) # Cap dt to avoid huge jumps on lags

        # If observed match current stable state, reset candidate and confidence
        if observed_state == self.current_state:
            self._candidate_state = observed_state
            self._confidence = 0.0
            return self.current_state

        # If observed state is different from our current candidate being built
        if observed_state != self._candidate_state:
            # Decay confidence instead of instant reset (Memory of recent events)
            self._confidence -= dt * 2.0  # Decays twice as fast as it builds
            if self._confidence <= 0.0:
                self._candidate_state = observed_state
                self._confidence = 0.0
            return self.current_state

        # We are observing the candidate state. Build confidence.
        self._confidence += dt
        
        # Asymmetric thresholds
        # If we are in 'default' and going to 'something else' -> use enter_sec
        # If we are in 'something else' and going to 'default' -> use exit_sec
        threshold = self.exit_sec if observed_state == self.default_state else self.enter_sec
        
        if self._confidence >= threshold:
            self.current_state = observed_state
            self._confidence = 0.0
            
        return self.current_state

class HysteresisManager:
    """
    Manages a collection of HysteresisState objects for various behavioral factors.
    """
    
    def __init__(self):
        self.factors: Dict[str, HysteresisState] = {
            "fatigue": HysteresisState("normal", enter_sec=2.5, exit_sec=6.0),
            "posture": HysteresisState("good", enter_sec=3.0, exit_sec=10.0),
            "phone": HysteresisState("not_detected", enter_sec=2.0, exit_sec=5.0),
            "distraction": HysteresisState("focused", enter_sec=3.0, exit_sec=5.0)
        }

    def process(self, now: float, observations: Dict[str, str]) -> Dict[str, str]:
        results = {}
        for key, state_obj in self.factors.items():
            obs = observations.get(key, state_obj.default_state)
            results[key] = state_obj.update(now, obs)
        return results
