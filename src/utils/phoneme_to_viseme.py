# coding: utf-8

"""
Enhanced Multi-Dimensional Phoneme to Viseme mapping for LivePortrait
HYBRID VERSION: Real tracked data + Manual refinements
Updated with video analysis from test.mp4
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
    """Enhanced viseme set with REAL TRACKED + OPTIMIZED parameters"""

    PHONEME_TO_VISEME_PARAMS = {
        # Silence/Rest
        'SIL': VisemeParams(
            jaw_open=0.1, lip_width=0.5, lip_protrusion=0.1
        ),
        'SP': VisemeParams(
            jaw_open=0.1, lip_width=0.5, lip_protrusion=0.1
        ),

        # Bilabial stops - TRACKED DATA
        'M': VisemeParams(
            jaw_open=0.645,  # TRACKED (was 0.08)
            lip_width=1.000,  # TRACKED (was 0.5)
            lip_protrusion=0.12,  # KEPT (tracking failed)
            upper_lip_raise=0.05,  # ADJUSTED
            lower_lip_lower=0.05,  # ADJUSTED
            corner_pull_horizontal=-0.171,  # TRACKED
            corner_pull_back=0.116,  # TRACKED
            lip_tightness=0.677,  # TRACKED
            anticipation_factor=0.15
        ),
        'B': VisemeParams(
            jaw_open=0.466,  # TRACKED (was 0.12)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.08,  # KEPT
            upper_lip_raise=0.03,  # ADJUSTED
            lower_lip_lower=0.03,  # ADJUSTED
            corner_pull_horizontal=0.166,  # TRACKED
            corner_pull_back=0.230,  # TRACKED
            lip_tightness=0.767,  # TRACKED
            overshoot_factor=0.1
        ),
        'P': VisemeParams(
            jaw_open=0.466,  # Same as B
            lip_width=1.000,
            lip_protrusion=0.08,
            upper_lip_raise=0.03,
            lower_lip_lower=0.03,
            corner_pull_horizontal=0.166,
            corner_pull_back=0.230,
            lip_tightness=0.80,
            overshoot_factor=0.12
        ),

        # Labiodental - TRACKED DATA
        'F': VisemeParams(
            jaw_open=0.577,  # TRACKED (was 0.18)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,  # KEPT
            upper_lip_raise=0.08,  # KEPT
            lower_lip_lower=0.28,  # KEPT
            corner_pull_horizontal=-0.039,  # TRACKED
            corner_pull_back=0.113,  # TRACKED
            lip_tightness=0.712,  # TRACKED
            anticipation_factor=0.12
        ),
        'V': VisemeParams(
            jaw_open=0.577,  # Same as F
            lip_width=1.000,
            lip_protrusion=0.0,
            upper_lip_raise=0.08,
            lower_lip_lower=0.32,
            corner_pull_horizontal=-0.039,
            corner_pull_back=0.113,
            lip_tightness=0.70,
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

        # Alveolar - TRACKED DATA
        'T': VisemeParams(
            jaw_open=0.444,  # TRACKED (was 0.28)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.05,  # ADJUSTED
            lower_lip_lower=0.05,  # ADJUSTED
            corner_pull_horizontal=0.110,  # TRACKED
            corner_pull_back=0.178,  # TRACKED
            lip_tightness=0.778,  # TRACKED
            overshoot_factor=0.1
        ),
        'D': VisemeParams(
            jaw_open=0.495,  # TRACKED (was 0.30)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.05,
            lower_lip_lower=0.05,
            corner_pull_horizontal=-0.110,  # TRACKED
            corner_pull_back=0.176,  # TRACKED
            lip_tightness=0.752,  # TRACKED
            overshoot_factor=0.1
        ),
        'S': VisemeParams(
            jaw_open=0.495,  # TRACKED (was 0.22)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.05,
            lower_lip_lower=0.05,
            corner_pull_horizontal=-0.094,  # TRACKED
            corner_pull_back=0.168,  # TRACKED
            lip_tightness=0.753  # TRACKED
        ),
        'Z': VisemeParams(
            jaw_open=0.495,  # Same as S
            lip_width=1.000,
            lip_protrusion=0.0,
            upper_lip_raise=0.05,
            lower_lip_lower=0.05,
            corner_pull_horizontal=-0.094,
            corner_pull_back=0.168,
            lip_tightness=0.753
        ),
        'N': VisemeParams(
            jaw_open=0.488,  # TRACKED (was 0.20)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.03,
            lower_lip_lower=0.03,
            corner_pull_horizontal=0.068,  # TRACKED
            corner_pull_back=0.165,  # TRACKED
            lip_tightness=0.756  # TRACKED
        ),
        'L': VisemeParams(
            jaw_open=0.497,  # TRACKED (was 0.30)
            lip_width=0.996,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.05,
            lower_lip_lower=0.05,
            corner_pull_horizontal=0.252,  # TRACKED
            corner_pull_back=0.160,  # TRACKED
            lip_tightness=0.751  # TRACKED
        ),

        # Postalveolar - KEPT MANUAL (not in tracked data)
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

        # Velar/Glottal - TRACKED DATA
        'K': VisemeParams(
            jaw_open=0.509,  # TRACKED (was 0.25)
            lip_width=0.992,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.03,
            lower_lip_lower=0.03,
            corner_pull_horizontal=-0.314,  # TRACKED
            corner_pull_back=0.108,  # TRACKED
            lip_tightness=0.745,  # TRACKED
            overshoot_factor=0.1
        ),
        'G': VisemeParams(
            jaw_open=0.509,  # Same as K
            lip_width=0.992,
            lip_protrusion=0.0,
            upper_lip_raise=0.03,
            lower_lip_lower=0.03,
            corner_pull_horizontal=-0.314,
            corner_pull_back=0.108,
            lip_tightness=0.745,
            overshoot_factor=0.1
        ),
        'NG': VisemeParams(
            jaw_open=0.660,  # TRACKED (was 0.22)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.03,
            lower_lip_lower=0.03,
            corner_pull_horizontal=0.539,  # TRACKED
            corner_pull_back=0.071,  # TRACKED
            lip_tightness=0.670  # TRACKED
        ),
        'H': VisemeParams(
            jaw_open=0.30, lip_width=0.6, lip_protrusion=0.0
        ),
        'Y': VisemeParams(
            jaw_open=0.445,  # TRACKED (was 0.22)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.05,
            lower_lip_lower=0.05,
            corner_pull_horizontal=-0.010,  # TRACKED
            corner_pull_back=0.165,  # TRACKED
            lip_tightness=0.777,  # TRACKED
            anticipation_factor=0.10
        ),

        # Approximants - TRACKED DATA
        'W': VisemeParams(
            jaw_open=0.364,  # TRACKED (was 0.25)
            lip_width=1.000,  # TRACKED (was 0.20!)
            lip_protrusion=0.75,  # KEPT (tracking failed)
            upper_lip_raise=0.0,
            lower_lip_lower=0.0,
            corner_pull_horizontal=-0.028,  # TRACKED
            corner_pull_back=0.085,  # TRACKED
            lip_tightness=0.818,  # TRACKED
            anticipation_factor=0.25
        ),
        'R': VisemeParams(
            jaw_open=0.755,  # TRACKED (was 0.35)
            lip_width=0.998,  # TRACKED (was 0.30)
            lip_protrusion=0.45,  # KEPT
            upper_lip_raise=0.0,
            lower_lip_lower=0.0,
            corner_pull_horizontal=-0.343,  # TRACKED
            corner_pull_back=0.109,  # TRACKED
            lip_tightness=0.622  # TRACKED
        ),
        'ER': VisemeParams(
            jaw_open=0.397,  # TRACKED (was 0.35)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.42,  # KEPT
            upper_lip_raise=0.0,
            lower_lip_lower=0.0,
            corner_pull_horizontal=-0.162,  # TRACKED
            corner_pull_back=0.189,  # TRACKED
            lip_tightness=0.802  # TRACKED
        ),

        # Front Vowels - TRACKED DATA
        'IY': VisemeParams(  # "ee"
            jaw_open=0.460,  # TRACKED (was 0.35)
            lip_width=1.000,  # TRACKED (was 0.90)
            lip_protrusion=0.0,
            upper_lip_raise=0.15,  # ADJUSTED
            lower_lip_lower=0.05,  # ADJUSTED
            corner_pull_horizontal=-0.223,  # TRACKED (was 0.50!)
            corner_pull_back=0.187,  # TRACKED
            lip_tightness=0.770  # TRACKED
        ),
        'IH': VisemeParams(  # "i"
            jaw_open=0.460,  # Same as IY
            lip_width=1.000,
            lip_protrusion=0.0,
            upper_lip_raise=0.12,
            lower_lip_lower=0.05,
            corner_pull_horizontal=-0.223,
            corner_pull_back=0.187,
            lip_tightness=0.770
        ),
        'EY': VisemeParams(  # "ay"
            jaw_open=0.628,  # TRACKED (was 0.50)
            lip_width=1.000,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.10,
            lower_lip_lower=0.08,
            corner_pull_horizontal=0.166,  # TRACKED
            corner_pull_back=0.141,  # TRACKED
            lip_tightness=0.686  # TRACKED
        ),
        'EH': VisemeParams(  # "e"
            jaw_open=0.628,  # Same as EY
            lip_width=1.000,
            lip_protrusion=0.0,
            upper_lip_raise=0.10,
            lower_lip_lower=0.08,
            corner_pull_horizontal=0.166,
            corner_pull_back=0.141,
            lip_tightness=0.686
        ),
        'AE': VisemeParams(  # "a"
            jaw_open=0.68, lip_width=0.80, lip_protrusion=0.0,
            corner_pull_horizontal=0.30, corner_pull_back=0.18,
            lower_lip_lower=0.15
        ),

        # Central Vowels - TRACKED DATA
        'AH': VisemeParams(  # "uh"
            jaw_open=0.509,  # TRACKED (was 0.70)
            lip_width=0.999,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.05,
            lower_lip_lower=0.05,
            corner_pull_horizontal=0.062,  # TRACKED
            corner_pull_back=0.150,  # TRACKED
            lip_tightness=0.746  # TRACKED
        ),
        'UH': VisemeParams(  # "oo" (book)
            jaw_open=0.523,  # TRACKED (was 0.45)
            lip_width=0.958,  # TRACKED (was 0.30)
            lip_protrusion=0.52,  # KEPT
            upper_lip_raise=0.0,
            lower_lip_lower=0.0,
            corner_pull_horizontal=-0.421,  # TRACKED
            corner_pull_back=0.000,  # TRACKED
            lip_tightness=0.738  # TRACKED
        ),

        # Back Vowels - KEPT MANUAL for AA, TRACKED for others
        'AA': VisemeParams(  # "ah" - KEPT MANUAL (not in tracked data)
            jaw_open=0.95, lip_width=0.55, lip_protrusion=0.0,
            corner_pull_horizontal=0.0, lower_lip_lower=0.28,
            lip_tightness=0.25, overshoot_factor=0.15
        ),
        'AO': VisemeParams(  # "aw"
            jaw_open=0.75, lip_width=0.30, lip_protrusion=0.55,
            corner_pull_horizontal=-0.15, lip_tightness=0.55
        ),
        'OW': VisemeParams(  # "o"
            jaw_open=0.477,  # TRACKED (was 0.60)
            lip_width=0.999,  # TRACKED (was 0.20!)
            lip_protrusion=0.78,  # KEPT (tracking failed)
            upper_lip_raise=0.0,
            lower_lip_lower=0.0,
            corner_pull_horizontal=-0.356,  # TRACKED
            corner_pull_back=0.143,  # TRACKED
            lip_tightness=0.762,  # TRACKED
            anticipation_factor=0.20
        ),
        'OY': VisemeParams(  # "oy"
            jaw_open=0.60, lip_width=0.25, lip_protrusion=0.68,
            corner_pull_horizontal=-0.20
        ),
        'UW': VisemeParams(  # "oo" (food)
            jaw_open=0.442,  # TRACKED (was 0.45)
            lip_width=1.000,  # TRACKED (was 0.15!)
            lip_protrusion=0.85,  # KEPT (tracking failed)
            upper_lip_raise=0.0,
            lower_lip_lower=0.0,
            corner_pull_horizontal=-0.027,  # TRACKED
            corner_pull_back=0.145,  # TRACKED
            lip_tightness=0.779,  # TRACKED
            anticipation_factor=0.22
        ),

        # Diphthongs
        'AW': VisemeParams(  # "ow" (cow)
            jaw_open=0.80, lip_width=0.35, lip_protrusion=0.48,
            corner_pull_horizontal=0.0
        ),
        'AY': VisemeParams(  # "eye"
            jaw_open=0.581,  # TRACKED (was 0.75)
            lip_width=0.993,  # TRACKED
            lip_protrusion=0.0,
            upper_lip_raise=0.08,
            lower_lip_lower=0.08,
            corner_pull_horizontal=0.077,  # TRACKED
            corner_pull_back=0.091,  # TRACKED
            lip_tightness=0.710  # TRACKED
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
