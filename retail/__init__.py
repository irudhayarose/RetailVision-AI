"""Retail occupancy system: identity tracking with verified entry/exit events."""

from .state import Event, State
from .tracks import PersonTrack, TrackManager

__all__ = ["Event", "State", "PersonTrack", "TrackManager"]
