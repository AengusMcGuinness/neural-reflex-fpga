"""Shared recording and decoder settings. Collection and training both read these."""

SAMPLE_RATE_HZ = 2048
CHANNELS = 8
SAMPLES_PER_BATCH = 16
SAMPLE_INTERVAL_US = 488.28125
WINDOW_MS = 200
HOP_MS = 20
LAGS_MS = (0, 80, 160, 240)
LABEL_DELAY_S = 0.08
BANDPASS_HZ = (20.0, 450.0)
NOTCH_HZ = 60.0
NOTCH_HARMONICS = (1, 2, 3)
NOTCH_Q = 30.0
MIN_CONFIDENCE = 0.5
MAX_POSE_GAP_S = 0.1

JOINT_NAMES = (
    "index_mcp", "index_pip",
    "middle_mcp", "middle_pip",
    "ring_mcp", "ring_pip",
    "pinky_mcp", "pinky_pip",
    "thumb_mcp", "thumb_ip",
    "thumb_opposition", "finger_spread",
    "palm_pitch", "palm_yaw",
)

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
