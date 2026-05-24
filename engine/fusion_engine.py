"""
FusionEngine - Level 3 Refactor
Standardizes global state derivation from independent sub-states.
Ensures deterministic priority-based state switching.
"""

class FusionEngine:
    """
    Combines sub-states from L2/L3 analyzers into a single, coherent global state.
    Follows a strict priority hierarchy to avoid state oscillation and inconsistent reporting.
    """
    
    # Priority: drowsy > phone > social > distracted > fatigued > slightly_fatigued > reading/writing > posture > focused
    PRIORITY_MAP = {
        "focused":             0,
        "posture_issue":       1,
        "reading":             2,
        "writing":             2,
        "slightly_distracted": 3,
        "distracted":          4,
        "slightly_fatigued":   5,
        "social_distraction":  6,
        "fatigued":            7,
        "phone_distraction":   8,
        "drowsy":              9
    }

    @staticmethod
    def compute_global_state(sub_states: list[str], scores: dict) -> str:
        """
        Drives the global state from a list of active sub-state labels.
        
        Args:
            sub_states: list of detected sub-state labels (e.g. ['poor_posture', 'fatigue_high'])
            scores: dict containing signal values (used for secondary tie-breaking if needed)
            
        Returns:
            The highest priority global state string.
        """
        if not sub_states:
            return "focused"

        # Map raw sub-states to priority-system states
        normalized_states = []
        for s in sub_states:
            s = s.lower()
            
            # Fatigue Mapping (New 4-tier logic)
            if "drowsy" in s or "microsleep" in s or "fatigue_high" in s:
                normalized_states.append("drowsy")
            elif "slightly_fatigued" in s:
                normalized_states.append("slightly_fatigued")
            elif "fatigued" in s or "fatigue_warning" in s:
                normalized_states.append("fatigued")
            
            # Phone Mapping
            elif "phone" in s:
                normalized_states.append("phone_distraction")
            
            # Social Mapping
            elif "social" in s or "person" in s or "interaction" in s:
                normalized_states.append("social_distraction")
            
            # Distraction Mapping
            elif "slightly_distracted" in s:
                normalized_states.append("slightly_distracted")
            elif "distracted" in s:
                normalized_states.append("distracted")
            
            # Task Mapping
            elif "reading" in s:
                normalized_states.append("reading")
            elif "writing" in s:
                normalized_states.append("writing")
            
            # Posture Mapping
            elif "posture" in s or "slouch" in s:
                normalized_states.append("posture_issue")
            
            # Legacy Distraction Mapping
            elif "gaze_away" in s:
                normalized_states.append("distracted") 

        if not normalized_states:
            return "focused"

        # Select state with highest priority value
        # Deterministic: if two states have same priority, max() picks one (lexicographical or first)
        best_state = "focused"
        max_priority = -1
        
        for state in normalized_states:
            prio = FusionEngine.PRIORITY_MAP.get(state, 0)
            if prio > max_priority:
                max_priority = prio
                best_state = state
                
        return best_state
