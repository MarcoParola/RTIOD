from abc import ABC, abstractmethod
from typing import List, Dict
import numpy as np

class BaseTracker(ABC):
    """ Base class for all tracking algorithms """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        """Reset internal tracker state (new sequence)"""
        pass

    @abstractmethod
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Args: detections: list of dicts with keys
            - bbox: [x, y, w, h]
            - score
            - class

        Returns: tracks: list of dicts with keys
            - id
            - bbox
        """
        pass

def get_tracker(tracker_name: str):

    if tracker_name == 'identity':
        from .identity_tracker import IdentityTracker
        return IdentityTracker()

    elif tracker_name == 'iou':
        from .iou_tracker import IOUTracker
        return IOUTracker()

    elif tracker_name == 'sort':
        from .sort_tracker import SORTTracker
        return SORTTracker()

    elif tracker_name == 'bytetrack':
        from .bytetrack_tracker import ByteTrackTracker
        return ByteTrackTracker()

    elif tracker_name == 'ocsort':
        from .ocsort_tracker import OCSORTTracker
        return OCSORTTracker()

    elif tracker_name == 'deepsort':
        from .deepsort_tracker import DeepSORTTracker
        return DeepSORTTracker()

    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")
