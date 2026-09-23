"""Live Taraweeh adaptation: dereverberation, pace normalization, breath/fatigue handling."""

from app.taraweeh_adapter.dereverb import EnvironmentProfile, adapt_acoustics, environment_profile, estimate_rt60
from app.taraweeh_adapter.fatigue_detector import detect_pauses, dynamic_stops, fatigue_report, pitch_profile
from app.taraweeh_adapter.pace_normalizer import build_local_tempo, classify_pace, hadr_to_tahqeeq_ratio

__all__ = [
    "EnvironmentProfile",
    "adapt_acoustics",
    "build_local_tempo",
    "classify_pace",
    "detect_pauses",
    "dynamic_stops",
    "environment_profile",
    "estimate_rt60",
    "fatigue_report",
    "hadr_to_tahqeeq_ratio",
    "pitch_profile",
]
