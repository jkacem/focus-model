import time
from typing import Optional, Dict

from config.cv_config import config
from engine.models import ConsolidatedStates, AlertStatus

class AlertManager:
    """
    Level 4: Temporal Validation & Alert Policy.
    Standardizes state persistence to prevent flickering and alert spam.
    """

    def __init__(self):
        # Tracking durations for each potential issue state
        self._persistence_timers: Dict[str, float] = {}
        self._validated_states: Dict[str, bool] = {}
        self._last_fired_times: Dict[str, float] = {}

    def _update_persistence(self, key: str, is_active: bool, now: float, required_duration: float) -> tuple[bool, float]:
        """
        Updates the persistence timer for a state and returns (is_validated, duration).
        Resets precisely when the state disappears.
        """
        if not is_active:
            self._persistence_timers.pop(key, None)
            self._validated_states[key] = False
            return False, 0.0

        # If it's the first time seeing this state, start the timer
        if key not in self._persistence_timers:
            self._persistence_timers[key] = now

        # Calculate how long it's been active
        elapsed = now - self._persistence_timers[key]
        
        # State becomes 'validated' only after required_duration
        is_validated = elapsed >= required_duration
        self._validated_states[key] = is_validated
        
        return is_validated, elapsed

    def _is_on_cooldown(self, key: str, now: float, cooldown_period: float) -> bool:
        """Checks if a specific alert type is currently on cooldown."""
        last_fired = self._last_fired_times.get(key, 0.0)
        return (now - last_fired) < cooldown_period

    def evaluate(self, state: ConsolidatedStates) -> AlertStatus:
        """
        Evaluates the current consolidated state and determines if a validated alert should fire.
        Returns the highest priority validated alert that is not on cooldown.
        """
        now = time.time()
        
        # Define alert types, their validation durations, and cooldowns
        # Priority: Fatigue High > Phone > Social > Fatigue Warning > Distraction > Posture
        alert_configs = [
            {
                "key": "fatigue_critical",
                "condition": state.fatigue_state == "fatigue_high",
                "val_duration": 2.5,        # Must be sustained for 2.5s
                "cooldown": 300.0,          # 5 min cooldown for high priority
                "level": "high",
                "reason": "Fatigue critique : Faites une pause immédiatement."
            },
            {
                "key": "phone_distraction",
                "condition": state.work_mode == "phone_distraction",
                "val_duration": config.ALERT_DELAY_PHONE,
                "cooldown": config.ALERT_COOLDOWN_SECONDS,
                "level": "high",
                "reason": "Attention : Utilisation du téléphone détectée."
            },
            {
                "key": "social_distraction",
                "condition": state.work_mode == "social_distraction",
                "val_duration": config.ALERT_DELAY_SOCIAL,
                "cooldown": config.ALERT_COOLDOWN_SECONDS,
                "level": "medium",
                "reason": "Note : Interaction sociale prolongée."
            },
            {
                "key": "fatigue_warning",
                "condition": state.fatigue_state == "fatigue_warning",
                "val_duration": config.ALERT_DELAY_FATIGUE,
                "cooldown": 120.0,          # 2 min cooldown
                "level": "medium",
                "reason": "Attention : Signes de fatigue détectés."
            },
            {
                "key": "distraction",
                "condition": state.work_mode == "brief_off_task",
                "val_duration": config.ALERT_DELAY_DISTRACTED,
                "cooldown": config.ALERT_COOLDOWN_SECONDS,
                "level": "low",
                "reason": "Rappel : Restez concentré sur votre tâche."
            },
            {
                "key": "posture",
                "condition": state.posture_state == "poor_persistent",
                "val_duration": config.ALERT_DELAY_POSTURE,
                "cooldown": 180.0,          # 3 min cooldown
                "level": "low",
                "reason": "Posture : Pensez à vous redresser."
            }
        ]

        # Global cooldown check (ensure user isn't bombarded with different alerts too close together)
        last_any_alert = max(self._last_fired_times.values()) if self._last_fired_times else 0.0
        if (now - last_any_alert) < 5.0:  # Absolute minimum 5s between any two alerts
            return AlertStatus(should_alert=False)

        for cfg in alert_configs:
            # 1. Update temporal validation
            is_validated, duration = self._update_persistence(
                cfg["key"], 
                cfg["condition"], 
                now, 
                cfg["val_duration"]
            )
            
            # 2. Check if validated and not on cooldown
            if is_validated and not self._is_on_cooldown(cfg["key"], now, cfg["cooldown"]):
                self._last_fired_times[cfg["key"]] = now
                return AlertStatus(
                    should_alert=True, 
                    alert_type=cfg["key"],
                    severity=cfg["level"], 
                    reason=cfg["reason"],
                    confidence=0.9, # Fixed high confidence for validated alerts
                    duration=round(duration, 2)
                )

        return AlertStatus(should_alert=False)
