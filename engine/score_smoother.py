from collections import deque
from typing import Dict, Any, List

class ScoreSmoother:
    """
    Implements a sliding window (moving average) for score smoothing.
    Used to reduce noise and instability in per-frame analyzer outputs.
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        # Map of score keys to their respective deques
        self._windows: Dict[str, deque] = {}

    def smooth(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Takes a dictionary of scores/metrics and returns the smoothed version.
        Non-numeric values are passed through unchanged.
        """
        smoothed_data = {}
        
        for key, value in data.items():
            # Only smooth numeric values (int or float)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in self._windows:
                    self._windows[key] = deque(maxlen=self.window_size)
                
                self._windows[key].append(float(value))
                
                # Compute moving average
                avg = sum(self._windows[key]) / len(self._windows[key])
                smoothed_data[key] = round(avg, 3)
            else:
                # Pass through non-numeric (bools, strings, lists, dicts)
                smoothed_data[key] = value
                
        return smoothed_data

    def reset(self, key: str = None):
        """Resets the window for a specific key or all keys."""
        if key:
            if key in self._windows:
                self._windows[key].clear()
        else:
            self._windows.clear()
