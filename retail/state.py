"""Explicit states and event types for the occupancy state machine."""
from enum import Enum


class State(str, Enum):
    APPROACHING_ENTRY = "APPROACHING_ENTRY"
    ENTERED = "ENTERED"
    INSIDE = "INSIDE"
    OCCLUDED = "OCCLUDED"
    POSSIBLY_EXITED = "POSSIBLY_EXITED"
    EXIT_CONFIRMED = "EXIT_CONFIRMED"
    REAPPEARED = "REAPPEARED"
    UNKNOWN = "UNKNOWN"


class Event(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    REENTRY = "REENTRY"
    REAPPEARED = "REAPPEARED"
    UNCERTAIN = "UNCERTAIN"
