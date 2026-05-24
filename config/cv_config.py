"""
Smart Focus - Centralized CV Configuration

Holds thresholds, window sizes, and hysteresis parameters for the entire pipeline.
Frames refer to steps in the main loop (effectively dt per iteration).
"""

class CVConfig:
    # Target pipeline FPS (used for converting seconds to frames if needed)
    TARGET_FPS = 30.0

    # =========================================================================
    # LEVEL 1: Analyzers (Instantaneous threshold tuning)
    # =========================================================================

    # Attention / Gaze
    YAW_DISTRACT_THRESH_DEG   = 35.0   # More tolerant (was 30)
    YAW_TOLERANCE_CENTER      = 18.0   # Tolerate natural head movements
    YAW_READING_THRESH_DEG    = 18.0
    PITCH_DOWN_THRESH_DEG     = 20.0
    PITCH_UP_THRESH_DEG       = 24.0
    LIP_MAR_SPEECH_THRESH     = 0.050
    SPEECH_MAR_STRONG_THRESH  = 0.080
    SPEECH_MAX_JITTER         = 0.008
    SOCIAL_GAZE_YAW_MIN_DEG   = 12.0

    # Fatigue
    MICROSLEEP_SECONDS        = 2.0    # FASTER trigger
    SLOW_BLINK_SECONDS        = 0.45   # FASTER trigger
    YAWN_MAR_BASE_OFFSET      = 0.35   # Raised: higher absolute MAR threshold (aligns with ratio=1.4)
    YAWN_SUSTAIN_SECONDS      = 0.3    # FASTER
    YAWN_COOLDOWN_SECONDS     = 2.0
    HEAD_BOB_THRESHOLD        = 0.03
    HEAD_BOB_WINDOW_SEC       = 1.2
    PERCLOS_WINDOW_SECONDS    = 60.0
    YAWN_FREQ_WINDOW_SEC      = 120.0  # Longer memory for yawning
    YAWN_FREQ_MIN_COUNT       = 2      # 2 yawns = frequent yawn
    READING_PITCH_DELTA       = 0.016  # normalized nose-eye delta for head-down compensation
    READING_EAR_RELAXATION    = 0.25   # relative EAR relaxation when reading downwards
    EYE_DROWSY_RELAXATION     = 0.15
    EYE_DROWSY_SECONDS        = 0.35
    SLEEPY_EAR_BONUS          = 15.0   # High bonus for drooping eyes
    YAWN_FATIGUE_BONUS        = 40.0   # 2 yawns = Fatigue High (>75)

    # Fatigue Factors (Weights)
    WEIGHT_FATIGUE_EYES       = 0.50   # EAR, PERCLOS, Slow blinks
    WEIGHT_FATIGUE_YAWN       = 0.20   # Yawning frequency
    WEIGHT_FATIGUE_POSTURE    = 0.15   # Relapsed posture
    WEIGHT_FATIGUE_BEHAVIOR   = 0.15   # Hands on knees, immobility
    
    # State Thresholds
    SCORE_FATIGUE_DROWSY      = 85     # Critical
    SCORE_FATIGUE_HEAVY       = 65     # Fatigued
    SCORE_FATIGUE_LIGHT       = 40     # Slightly Fatigued

    # Posture
    SLOUCH_PENALTY_MULT       = 0.8
    TILT_PENALTY_MULT         = 2.0
    FWD_HEAD_PENALTY_MULT     = 1.5
    POSTURE_LATERAL_DEG_TOL   = 6.0
    POSTURE_FORWARD_DEG_TOL   = 10.0
    POSTURE_FORWARD_DEG_MAX   = 35.0
    POSTURE_LATERAL_DEG_MAX   = 25.0
    LEAN_PENALTY_MULT         = 1.2
    POSTURE_BAD_HOLD_SECONDS  = 2.5

    # Hand detection thresholds (normalized landmark coords)
    HAND_KNEE_Y_TOL           = 0.08   # vertical tolerance wrist-hip
    HAND_KNEE_X_TOL           = 0.10   # horizontal tolerance wrist-hip
    HAND_FACE_DIST_TOL        = 0.20   # max normalized distance wrist-nose

    # =========================================================================
    # LEVEL 2: Weak Hypotheses (Evidence Accumulation Weights)
    # =========================================================================
    
    # Reading Evidence
    WEIGHT_READING_PITCH_DOWN = 0.6
    WEIGHT_READING_YAW_CENTER = 0.3
    WEIGHT_READING_STABILITY  = 0.2

    # Writing Evidence
    WEIGHT_WRITING_PITCH_DOWN = 0.5
    WEIGHT_WRITING_HANDS_NEAR = 0.5

    # Thinking Evidence
    WEIGHT_THINKING_YAW_AWAY  = 0.5
    WEIGHT_THINKING_CALM      = 0.6

    # Self-Explaining Evidence
    WEIGHT_SPEECH_MAR_ACTIVE  = 0.55
    WEIGHT_SPEECH_SINGLE_PRES = 0.15
    SELF_EXPLAIN_DELAY        = 4.0

    # Phone Evidence
    WEIGHT_PHONE_DETECTED     = 0.8
    WEIGHT_PHONE_GAZE_MATCH   = 0.4

    # =========================================================================
    # LEVEL 3: Consolidated States (Transitions & Hysteresis)
    # =========================================================================
    
    # Transition Delays (seconds)
    TRANSITION_DELAY_FOCUSED    = 0.8
    TRANSITION_DELAY_READING    = 2.5
    TRANSITION_DELAY_WRITING    = 2.5
    TRANSITION_DELAY_THINKING   = 3.0
    TRANSITION_DELAY_DISTRACTED = 2.5
    TRANSITION_DELAY_SOCIAL     = 4.0
    TRANSITION_DELAY_PHONE      = 2.0    # Longer: needs sustained evidence (was 1.5)

    # Scores
    SCORE_FATIGUE_HIGH        = 80     # Equivalent to Drowsy
    SCORE_FATIGUE_WARNING     = 50     # Equivalent to Fatigued
    SCORE_POSTURE_BAD         = 30

    # EMA smoothing
    EMA_ALPHA_FATIGUE         = 0.08   # Slightly faster for fatigue accumulation
    EMA_ALPHA_FATIGUE_DECAY   = 0.015  # Slow decay for fatigue memory (hysteresis)
    EMA_ALPHA_POSTURE         = 0.10

    # Temporal smoothing / stability
    # Buffer length target: 30–60 frames (at 30fps, 2.0s ≈ 60 frames)
    SMOOTHING_WINDOW_SECONDS  = 2.5    # Increased for more stability
    MIN_STATE_DWELL_SECONDS   = 3.0    # 3 seconds minimum before major state change
    DISTRACTION_WINDOW_SEC    = 4.0    # Window to confirm deep distraction

    # =========================================================================
    # LEVEL 4: Alerts (Strict Conservative Policy)
    # =========================================================================

    ALERT_DELAY_PHONE         = 5.0
    ALERT_DELAY_SOCIAL        = 8.0
    ALERT_DELAY_DISTRACTED    = 12.0
    ALERT_DELAY_POSTURE       = 20.0
    ALERT_DELAY_FATIGUE       = 22.0

    # Prevent alert spam
    ALERT_COOLDOWN_SECONDS    = 5.0

config = CVConfig()
