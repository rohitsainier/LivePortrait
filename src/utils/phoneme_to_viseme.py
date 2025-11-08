# coding: utf-8

"""
Enhanced Multi-Dimensional Phoneme to Viseme mapping for LivePortrait
OPTIMIZED VERSION with improved parameters and temporal dynamics
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class VisemeParams:
    """Multi-dimensional viseme parameters for natural lip movement"""

    # Primary controls (0.0 to 1.0 normalized)
    jaw_open: float = 0.0       # Vertical mouth opening
    lip_width: float = 0.5      # Horizontal stretch
    lip_protrusion: float = 0.0 # Forward lip push

    # Secondary controls
    upper_lip_raise: float = 0.0
    lower_lip_lower: float = 0.0
    corner_pull_horizontal: float = 0.0
    corner_pull_back: float = 0.0

    # Fine controls
    lip_tightness: float = 0.5
    asymmetry_left_right: float = 0.0

    # NEW: Temporal dynamics
    anticipation_factor: float = 0.0  # Pre-movement hint
    overshoot_factor: float = 0.0     # Natural overshoot

    def to_dict(self) -> Dict[str, float]:
        return {
            'jaw_open': self.jaw_open,
            'lip_width': self.lip_width,
            'lip_protrusion': self.lip_protrusion,
            'upper_lip_raise': self.upper_lip_raise,
            'lower_lip_lower': self.lower_lip_lower,
            'corner_pull_horizontal': self.corner_pull_horizontal,
            'corner_pull_back': self.corner_pull_back,
            'lip_tightness': self.lip_tightness,
            'asymmetry_left_right': self.asymmetry_left_right,
            'anticipation_factor': self.anticipation_factor,
            'overshoot_factor': self.overshoot_factor,
        }


class EnhancedLivePortraitVisemes:
    """Enhanced viseme set with OPTIMIZED parameters for better lip-sync"""

    PHONEME_TO_VISEME_PARAMS = {
        # Silence/Rest
        'SIL': VisemeParams(
            jaw_open=0.1, lip_width=0.5, lip_protrusion=0.1
        ),
        'SP': VisemeParams(
            jaw_open=0.1, lip_width=0.5, lip_protrusion=0.1
        ),

        # Bilabial stops - INCREASED jaw_open for visibility
        'M': VisemeParams(
            jaw_open=0.08, lip_width=0.5, lip_protrusion=0.12,
            lip_tightness=0.7, anticipation_factor=0.15
        ),
        'B': VisemeParams(
            jaw_open=0.12, lip_width=0.5, lip_protrusion=0.08,
            lip_tightness=0.6, overshoot_factor=0.1
        ),
        'P': VisemeParams(
            jaw_open=0.12, lip_width=0.5, lip_protrusion=0.08,
            lip_tightness=0.8, overshoot_factor=0.12
        ),

        # Labiodental - ENHANCED for visibility
        'F': VisemeParams(
            jaw_open=0.18, lip_width=0.5, lip_protrusion=0.0,
            upper_lip_raise=0.08, lower_lip_lower=0.28,
            anticipation_factor=0.12
        ),
        'V': VisemeParams(
            jaw_open=0.22, lip_width=0.5, lip_protrusion=0.0,
            upper_lip_raise=0.08, lower_lip_lower=0.32,
            overshoot_factor=0.08
        ),

        # Dental
        'TH': VisemeParams(
            jaw_open=0.25, lip_width=0.6, lip_protrusion=0.0,
            upper_lip_raise=0.10, lower_lip_lower=0.10
        ),
        'DH': VisemeParams(
            jaw_open=0.25, lip_width=0.6, lip_protrusion=0.0,
            upper_lip_raise=0.10, lower_lip_lower=0.10
        ),

        # Alveolar - INCREASED visibility
        'T': VisemeParams(
            jaw_open=0.28, lip_width=0.65, lip_protrusion=0.0,
            corner_pull_horizontal=0.15, overshoot_factor=0.1
        ),
        'D': VisemeParams(
            jaw_open=0.30, lip_width=0.65, lip_protrusion=0.0,
            corner_pull_horizontal=0.15, overshoot_factor=0.1
        ),
        'S': VisemeParams(
            jaw_open=0.22, lip_width=0.70, lip_protrusion=0.0,
            corner_pull_horizontal=0.20, lip_tightness=0.75
        ),
        'Z': VisemeParams(
            jaw_open=0.22, lip_width=0.70, lip_protrusion=0.0,
            corner_pull_horizontal=0.20, lip_tightness=0.75
        ),
        'N': VisemeParams(
            jaw_open=0.20, lip_width=0.6, lip_protrusion=0.0
        ),
        'L': VisemeParams(
            jaw_open=0.30, lip_width=0.65, lip_protrusion=0.0,
            corner_pull_horizontal=0.18
        ),

        # Postalveolar - ENHANCED protrusion
        'SH': VisemeParams(
            jaw_open=0.28, lip_width=0.30, lip_protrusion=0.50,
            corner_pull_horizontal=-0.15, lip_tightness=0.65,
            anticipation_factor=0.18
        ),
        'ZH': VisemeParams(
            jaw_open=0.28, lip_width=0.30, lip_protrusion=0.50,
            corner_pull_horizontal=-0.15, lip_tightness=0.65
        ),
        'CH': VisemeParams(
            jaw_open=0.30, lip_width=0.30, lip_protrusion=0.42,
            corner_pull_horizontal=-0.10, overshoot_factor=0.12
        ),
        'JH': VisemeParams(
            jaw_open=0.32, lip_width=0.30, lip_protrusion=0.42,
            corner_pull_horizontal=-0.10, overshoot_factor=0.12
        ),

        # Velar/Glottal
        'K': VisemeParams(
            jaw_open=0.25, lip_width=0.6, lip_protrusion=0.0,
            overshoot_factor=0.1
        ),
        'G': VisemeParams(
            jaw_open=0.28, lip_width=0.6, lip_protrusion=0.0,
            overshoot_factor=0.1
        ),
        'NG': VisemeParams(
            jaw_open=0.22, lip_width=0.55, lip_protrusion=0.0
        ),
        'H': VisemeParams(
            jaw_open=0.30, lip_width=0.6, lip_protrusion=0.0
        ),
        'Y': VisemeParams(
            jaw_open=0.22, lip_width=0.75, lip_protrusion=0.0,
            corner_pull_horizontal=0.28, anticipation_factor=0.10
        ),

        # Approximants - ENHANCED rounding
        'W': VisemeParams(
            jaw_open=0.25, lip_width=0.20, lip_protrusion=0.75,
            corner_pull_horizontal=-0.22, lip_tightness=0.75,
            anticipation_factor=0.25
        ),
        'R': VisemeParams(
            jaw_open=0.35, lip_width=0.30, lip_protrusion=0.45,
            corner_pull_horizontal=-0.08
        ),
        'ER': VisemeParams(
            jaw_open=0.35, lip_width=0.30, lip_protrusion=0.42,
            corner_pull_horizontal=-0.08
        ),

        # Front Vowels - INCREASED jaw opening
        'IY': VisemeParams(  # "ee"
            jaw_open=0.35, lip_width=0.90, lip_protrusion=0.0,
            corner_pull_horizontal=0.50, corner_pull_back=0.30,
            lip_tightness=0.65
        ),
        'IH': VisemeParams(  # "i"
            jaw_open=0.40, lip_width=0.80, lip_protrusion=0.0,
            corner_pull_horizontal=0.38, corner_pull_back=0.20
        ),
        'EY': VisemeParams(  # "ay"
            jaw_open=0.50, lip_width=0.80, lip_protrusion=0.0,
            corner_pull_horizontal=0.32, corner_pull_back=0.18
        ),
        'EH': VisemeParams(  # "e"
            jaw_open=0.55, lip_width=0.75, lip_protrusion=0.0,
            corner_pull_horizontal=0.25, corner_pull_back=0.12
        ),
        'AE': VisemeParams(  # "a"
            jaw_open=0.68, lip_width=0.80, lip_protrusion=0.0,
            corner_pull_horizontal=0.30, corner_pull_back=0.18,
            lower_lip_lower=0.15
        ),

        # Central Vowels
        'AH': VisemeParams(  # "uh"
            jaw_open=0.70, lip_width=0.60, lip_protrusion=0.0,
            corner_pull_horizontal=0.08
        ),
        'UH': VisemeParams(  # "oo" (book)
            jaw_open=0.45, lip_width=0.30, lip_protrusion=0.52,
            corner_pull_horizontal=-0.15
        ),

        # Back Vowels - MAXIMUM opening for "AA"
        'AA': VisemeParams(  # "ah" - CRITICAL FOR VISIBILITY
            jaw_open=0.95, lip_width=0.55, lip_protrusion=0.0,
            corner_pull_horizontal=0.0, lower_lip_lower=0.28,
            lip_tightness=0.25, overshoot_factor=0.15
        ),
        'AO': VisemeParams(  # "aw"
            jaw_open=0.75, lip_width=0.30, lip_protrusion=0.55,
            corner_pull_horizontal=-0.15, lip_tightness=0.55
        ),
        'OW': VisemeParams(  # "o"
            jaw_open=0.60, lip_width=0.20, lip_protrusion=0.78,
            corner_pull_horizontal=-0.28, lip_tightness=0.65,
            anticipation_factor=0.20
        ),
        'OY': VisemeParams(  # "oy"
            jaw_open=0.60, lip_width=0.25, lip_protrusion=0.68,
            corner_pull_horizontal=-0.20
        ),
        'UW': VisemeParams(  # "oo" (food)
            jaw_open=0.45, lip_width=0.15, lip_protrusion=0.85,
            corner_pull_horizontal=-0.32, lip_tightness=0.75,
            anticipation_factor=0.22
        ),

        # Diphthongs
        'AW': VisemeParams(  # "ow" (cow)
            jaw_open=0.80, lip_width=0.35, lip_protrusion=0.48,
            corner_pull_horizontal=0.0
        ),
        'AY': VisemeParams(  # "eye"
            jaw_open=0.75, lip_width=0.65, lip_protrusion=0.0,
            corner_pull_horizontal=0.20
        ),
    }

    @classmethod
    def get_viseme_params(cls, phoneme: str) -> VisemeParams:
        """Get viseme parameters for a phoneme"""
        phoneme = phoneme.upper().replace('0', '').replace('1', '').replace('2', '')
        return cls.PHONEME_TO_VISEME_PARAMS.get(phoneme, VisemeParams())


class MultiDimensionalPhonemeMapper:
    """Enhanced phoneme mapper with temporal dynamics"""

    def __init__(self):
        pass

    def phoneme_to_viseme_params(self, phoneme: str) -> VisemeParams:
        return EnhancedLivePortraitVisemes.get_viseme_params(phoneme)

    def phoneme_sequence_to_viseme_timeline(
        self,
        phoneme_sequence: List[Tuple[str, float, float]]
    ) -> List[Tuple[VisemeParams, float, float]]:
        viseme_timeline = []
        for phoneme, start_time, end_time in phoneme_sequence:
            viseme_params = self.phoneme_to_viseme_params(phoneme)
            viseme_timeline.append((viseme_params, start_time, end_time))
        return viseme_timeline

    def interpolate_viseme_params(
        self,
        viseme_timeline: List[Tuple[VisemeParams, float, float]],
        fps: int = 25,
        total_duration: float = None,
        smoothing: bool = True,
        smoothing_window: int = 5,
        coarticulation_factor: float = 0.25
    ) -> List[Dict[str, float]]:
        """
        Interpolate with IMPROVED temporal dynamics
        """
        if total_duration is None:
            total_duration = max([end for _, _, end in viseme_timeline])

        num_frames = int(total_duration * fps)
        param_names = list(VisemeParams().to_dict().keys())
        param_arrays = {name: np.zeros(num_frames, dtype=np.float32) for name in param_names}

        # Fill base values with anticipation
        for frame_idx in range(num_frames):
            frame_time = frame_idx / fps

            current_params = None
            prev_params = None
            next_params = None
            current_idx = None

            for i, (viseme_params, start_time, end_time) in enumerate(viseme_timeline):
                if start_time <= frame_time < end_time:
                    current_params = viseme_params
                    current_idx = i
                    if i > 0:
                        prev_params = viseme_timeline[i-1][0]
                    if i < len(viseme_timeline) - 1:
                        next_params = viseme_timeline[i+1][0]
                    break

            if current_params is None:
                continue

            current_dict = current_params.to_dict()

            # Apply coarticulation + anticipation
            for param_name in param_names:
                base_value = current_dict[param_name]

                if coarticulation_factor > 0 and (prev_params or next_params):
                    neighbors_avg = base_value
                    neighbor_count = 1

                    if prev_params:
                        neighbors_avg += prev_params.to_dict()[param_name]
                        neighbor_count += 1
                    if next_params:
                        # Add anticipation: blend more heavily with next phoneme
                        next_weight = 1.0 + current_params.anticipation_factor
                        neighbors_avg += next_params.to_dict()[param_name] * next_weight
                        neighbor_count += next_weight

                    neighbors_avg /= neighbor_count
                    blended_value = (
                        base_value * (1 - coarticulation_factor) +
                        neighbors_avg * coarticulation_factor
                    )
                    param_arrays[param_name][frame_idx] = blended_value
                else:
                    param_arrays[param_name][frame_idx] = base_value

        # Apply smoothing
        if smoothing and smoothing_window > 1:
            for param_name in param_names:
                param_arrays[param_name] = self._moving_average(
                    param_arrays[param_name], smoothing_window
                )

        # Convert to list of dictionaries
        result = []
        for frame_idx in range(num_frames):
            frame_params = {
                param_name: float(param_arrays[param_name][frame_idx])
                for param_name in param_names
            }
            result.append(frame_params)

        return result

    @staticmethod
    def _moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        if window_size < 2:
            return data
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(data, kernel, mode='same')
        for i in range(window_size // 2):
            smoothed[i] = data[i]
            smoothed[-(i+1)] = data[-(i+1)]
        return smoothed
