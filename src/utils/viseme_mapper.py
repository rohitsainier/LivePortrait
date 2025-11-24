# coding: utf-8
"""
SIMPLIFIED Production-level Text-to-Viseme mapping system
Uses ONLY LivePortrait's retarget_lip function with single lip_open parameter
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

# Optional dependencies for enhanced functionality
try:
    import phonemizer
    PHONEMIZER_AVAILABLE = True
except ImportError:
    PHONEMIZER_AVAILABLE = False
    print("Warning: phonemizer not installed. Install with: pip install phonemizer")

try:
    from pydub import AudioSegment
    import librosa
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: audio libraries not installed. Install with: pip install pydub librosa")


class VisemeType(Enum):
    """Standard viseme types based on Disney/Preston Blair viseme set"""
    SILENCE = "sil"      # Silence/Rest
    PP_BB_MM = "PP"      # p, b, m
    F_V = "FF"           # f, v
    TH = "TH"            # th (thin, then)
    DD = "DD"            # t, d
    KK = "kk"            # k, g
    CH_JJ_SH = "CH"      # ch, j, sh
    SS = "SS"            # s, z
    NN_NN = "nn"         # n, ng
    RR = "RR"            # r
    AA = "aa"            # a (had)
    E = "E"              # e (bed)
    I = "I"              # i (tip)
    O = "O"              # o (go)
    U = "U"              # u (boot)
    AI = "AI"            # ai (bite)
    EI = "EI"            # ei (bait)
    OW = "OW"            # ow (show)
    AW = "AW"            # aw (how)
    L = "L"              # l
    WQ = "WQ"            # w, q
    FV = "FV"            # Wider f, v
    ER = "ER"            # er (her)


@dataclass
class VisemeFrame:
    """Represents a single viseme with timing information"""
    viseme: VisemeType
    start_time: float  # in seconds
    duration: float    # in seconds
    intensity: float = 1.0  # 0.0 to 1.0
    phoneme: str = ""

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


@dataclass
class LipParameters:
    """SIMPLIFIED: Only lip_open parameter for LivePortrait's retarget_lip"""
    lip_open: float = 0.0  # 0.0 (closed) to 0.8 (wide open)

    def to_retarget_input(self) -> float:
        """Convert to retarget_lip input (just returns lip_open)"""
        return self.lip_open


class VisemeToLipMapper:
    """Maps visemes to lip_open values (0.0 to 0.8)"""

    # SIMPLIFIED: Viseme to lip_open mapping (hand-tuned for realism)
    VISEME_LIP_MAP: Dict[VisemeType, float] = {
        # Silence
        VisemeType.SILENCE: 0.0,

        # Consonants - Bilabials (lips closed)
        VisemeType.PP_BB_MM: 0.0,

        # Consonants - Labiodentals (slight opening)
        VisemeType.F_V: 0.15,
        VisemeType.FV: 0.18,

        # Consonants - Dentals (slight opening with teeth)
        VisemeType.TH: 0.20,

        # Consonants - Alveolars (small opening)
        VisemeType.DD: 0.25,
        VisemeType.SS: 0.15,
        VisemeType.NN_NN: 0.20,
        VisemeType.L: 0.25,

        # Consonants - Velars (medium opening)
        VisemeType.KK: 0.30,

        # Consonants - Postalveolars (small-medium opening)
        VisemeType.CH_JJ_SH: 0.20,

        # Consonants - Approximants
        VisemeType.RR: 0.25,
        VisemeType.WQ: 0.20,

        # Vowels - Front (varying openness)
        VisemeType.I: 0.20,      # "sit" - small opening, wide
        VisemeType.E: 0.35,      # "bed" - medium opening
        VisemeType.EI: 0.35,     # "bait" - medium opening

        # Vowels - Central (medium-large opening)
        VisemeType.ER: 0.30,     # "her" - medium opening
        VisemeType.AA: 0.60,     # "father" - WIDE opening

        # Vowels - Back (varying with rounding)
        VisemeType.U: 0.30,      # "boot" - medium, rounded
        VisemeType.O: 0.50,      # "go" - large, rounded

        # Diphthongs
        VisemeType.AI: 0.50,     # "bite" - starts wide
        VisemeType.OW: 0.45,     # "show" - medium-large, rounded
        VisemeType.AW: 0.60,     # "how" - wide opening
    }

    @classmethod
    def get_lip_open(cls, viseme: VisemeType, intensity: float = 1.0) -> float:
        """Get lip_open value for a viseme with intensity scaling"""
        base_value = cls.VISEME_LIP_MAP.get(viseme, 0.0)
        return base_value * intensity

    @classmethod
    def interpolate_lip_open(cls, value1: float, value2: float, alpha: float) -> float:
        """Linear interpolation between two lip_open values"""
        return value1 * (1 - alpha) + value2 * alpha


class PhonemeToVisemeMapper:
    """Maps phonemes (IPA or ARPABET) to visemes"""

    # IPA to Viseme mapping
    IPA_TO_VISEME: Dict[str, VisemeType] = {
        # Bilabials
        'p': VisemeType.PP_BB_MM, 'b': VisemeType.PP_BB_MM, 'm': VisemeType.PP_BB_MM,
        # Labiodentals
        'f': VisemeType.F_V, 'v': VisemeType.F_V,
        # Dentals
        'θ': VisemeType.TH, 'ð': VisemeType.TH,
        # Alveolars
        't': VisemeType.DD, 'd': VisemeType.DD, 'n': VisemeType.NN_NN,
        's': VisemeType.SS, 'z': VisemeType.SS,
        'l': VisemeType.L, 'r': VisemeType.RR,
        # Postalveolars
        'ʃ': VisemeType.CH_JJ_SH, 'ʒ': VisemeType.CH_JJ_SH,
        'tʃ': VisemeType.CH_JJ_SH, 'dʒ': VisemeType.CH_JJ_SH,
        # Velars
        'k': VisemeType.KK, 'g': VisemeType.KK, 'ŋ': VisemeType.NN_NN,
        # Glottals
        'h': VisemeType.SILENCE,
        # Approximants
        'w': VisemeType.WQ, 'j': VisemeType.I,
        # Vowels - Front
        'i': VisemeType.I, 'ɪ': VisemeType.I,
        'e': VisemeType.EI, 'ɛ': VisemeType.E,
        'æ': VisemeType.AA,
        # Vowels - Central
        'ə': VisemeType.ER, 'ʌ': VisemeType.ER, 'ɜ': VisemeType.ER,
        'ɐ': VisemeType.AA,
        # Vowels - Back
        'u': VisemeType.U, 'ʊ': VisemeType.U,
        'o': VisemeType.O, 'ɔ': VisemeType.O,
        'ɑ': VisemeType.AA, 'ɒ': VisemeType.O,
        # Diphthongs
        'aɪ': VisemeType.AI, 'eɪ': VisemeType.EI,
        'ɔɪ': VisemeType.OW, 'aʊ': VisemeType.AW,
        'oʊ': VisemeType.OW, 'ɪə': VisemeType.I,
        'ɛə': VisemeType.E, 'ʊə': VisemeType.U,
    }

    # ARPABET to Viseme mapping (for CMU dict compatibility)
    ARPABET_TO_VISEME: Dict[str, VisemeType] = {
        # Consonants
        'P': VisemeType.PP_BB_MM, 'B': VisemeType.PP_BB_MM, 'M': VisemeType.PP_BB_MM,
        'F': VisemeType.F_V, 'V': VisemeType.F_V,
        'TH': VisemeType.TH, 'DH': VisemeType.TH,
        'T': VisemeType.DD, 'D': VisemeType.DD, 'N': VisemeType.NN_NN,
        'S': VisemeType.SS, 'Z': VisemeType.SS,
        'L': VisemeType.L, 'R': VisemeType.RR,
        'SH': VisemeType.CH_JJ_SH, 'ZH': VisemeType.CH_JJ_SH,
        'CH': VisemeType.CH_JJ_SH, 'JH': VisemeType.CH_JJ_SH,
        'K': VisemeType.KK, 'G': VisemeType.KK, 'NG': VisemeType.NN_NN,
        'HH': VisemeType.SILENCE,
        'W': VisemeType.WQ, 'Y': VisemeType.I,
        # Vowels
        'IY': VisemeType.I, 'IH': VisemeType.I,
        'EY': VisemeType.EI, 'EH': VisemeType.E,
        'AE': VisemeType.AA,
        'AH': VisemeType.ER, 'ER': VisemeType.ER,
        'UW': VisemeType.U, 'UH': VisemeType.U,
        'OW': VisemeType.O, 'AO': VisemeType.O,
        'AA': VisemeType.AA,
        # Diphthongs
        'AY': VisemeType.AI, 'AW': VisemeType.AW,
        'OY': VisemeType.OW,
    }

    @classmethod
    def phoneme_to_viseme(cls, phoneme: str, notation: str = "ipa") -> VisemeType:
        """Convert a phoneme to viseme"""
        phoneme = phoneme.strip().lower() if notation == "ipa" else phoneme.strip().upper()

        mapping = cls.IPA_TO_VISEME if notation == "ipa" else cls.ARPABET_TO_VISEME

        # Try direct lookup
        if phoneme in mapping:
            return mapping[phoneme]

        # Try without stress markers (for ARPABET)
        phoneme_base = re.sub(r'\d', '', phoneme)
        if phoneme_base in mapping:
            return mapping[phoneme_base]

        # Default to silence
        return VisemeType.SILENCE


class TextToVisemeConverter:
    """Converts text to timed viseme sequence"""

    def __init__(self, language: str = "en-us", backend: str = "espeak"):
        """
        Args:
            language: Language code (e.g., 'en-us', 'es', 'fr')
            backend: Phonemizer backend ('espeak', 'espeak-mbrola', 'festival')
        """
        self.language = language
        self.backend = backend

        if PHONEMIZER_AVAILABLE:
            from phonemizer.backend import EspeakBackend
            from phonemizer.separator import Separator

            self.phonemizer = EspeakBackend(
                language=language,
                preserve_punctuation=True,
                with_stress=True
            )
            self.separator = Separator(phone=' ', word=' | ')
        else:
            self.phonemizer = None
            print("Warning: Phonemizer not available. Using fallback text processing.")

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme sequence"""
        if self.phonemizer is not None:
            phonemes = self.phonemizer.phonemize(
                [text],
                separator=self.separator,
                strip=True
            )[0]
            return [p for p in phonemes.split() if p != '|']
        else:
            # Fallback: simple character-based approximation
            return self._fallback_text_to_phonemes(text)

    def _fallback_text_to_phonemes(self, text: str) -> List[str]:
        """Fallback phoneme extraction (very basic)"""
        # Simple character to phoneme mapping
        char_map = {
            'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'ɔ', 'u': 'ʊ',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': 'h', 'j': 'dʒ', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'p': 'p', 'q': 'k', 'r': 'r', 's': 's',
            't': 't', 'v': 'v', 'w': 'w', 'x': 'ks', 'y': 'j', 'z': 'z'
        }

        phonemes = []
        for char in text.lower():
            if char in char_map:
                phonemes.append(char_map[char])
            elif char == ' ':
                phonemes.append('|')  # Word boundary

        return phonemes

    def text_to_visemes(self, text: str, duration: float = None,
                       words_per_minute: float = 150) -> List[VisemeFrame]:
        """
        Convert text to timed viseme sequence

        Args:
            text: Input text
            duration: Total duration in seconds (if None, estimated from WPM)
            words_per_minute: Speech rate for duration estimation

        Returns:
            List of VisemeFrame objects with timing
        """
        # Get phonemes
        phonemes = self.text_to_phonemes(text)

        # Estimate duration if not provided
        if duration is None:
            word_count = len(text.split())
            duration = (word_count / words_per_minute) * 60

        # Convert phonemes to visemes
        viseme_sequence = []
        for phoneme in phonemes:
            if phoneme == '|':  # Word boundary
                continue
            viseme = PhonemeToVisemeMapper.phoneme_to_viseme(phoneme, notation="ipa")
            viseme_sequence.append((viseme, phoneme))

        # Assign timing (equal duration for now, can be improved with audio alignment)
        if len(viseme_sequence) == 0:
            return [VisemeFrame(VisemeType.SILENCE, 0, duration)]

        frame_duration = duration / len(viseme_sequence)

        timed_visemes = []
        current_time = 0.0

        for viseme, phoneme in viseme_sequence:
            # Adjust duration based on phoneme type
            duration_mult = self._get_phoneme_duration_multiplier(phoneme)
            actual_duration = frame_duration * duration_mult

            timed_visemes.append(VisemeFrame(
                viseme=viseme,
                start_time=current_time,
                duration=actual_duration,
                phoneme=phoneme,
                intensity=1.0
            ))

            current_time += actual_duration

        # Normalize timing to fit exact duration
        total_time = sum(v.duration for v in timed_visemes)
        if total_time > 0:
            scale = duration / total_time
            current_time = 0.0
            for viseme_frame in timed_visemes:
                viseme_frame.duration *= scale
                viseme_frame.start_time = current_time
                current_time += viseme_frame.duration

        return timed_visemes

    def _get_phoneme_duration_multiplier(self, phoneme: str) -> float:
        """Get relative duration multiplier for different phoneme types"""
        # Vowels are typically longer
        vowels = set('aeiouəɛɪɔʊʌɜ')
        if any(v in phoneme.lower() for v in vowels):
            return 1.3

        # Stops are shorter
        stops = set('pbtdkg')
        if phoneme.lower() in stops:
            return 0.7

        return 1.0


class AudioToVisemeConverter:
    """Convert audio to viseme sequence using Whisper transcription"""

    def __init__(self, whisper_model: str = "base"):
        """
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        try:
            from .audio_processor import WhisperTranscriber, AudioVisemeAligner, WHISPER_AVAILABLE

            if not WHISPER_AVAILABLE:
                raise ImportError("Whisper not available")

            self.transcriber = WhisperTranscriber(model_size=whisper_model)
            self.aligner = AudioVisemeAligner()
            self.text_converter = TextToVisemeConverter()

        except ImportError as e:
            print(f"Warning: Audio processing not available: {e}")
            self.transcriber = None
            self.aligner = None
            self.text_converter = None

    def audio_to_visemes(
        self,
        audio_path: str,
        text: Optional[str] = None,
        language: Optional[str] = None
    ) -> List[VisemeFrame]:
        """
        Convert audio to viseme sequence with accurate timing

        Args:
            audio_path: Path to audio file
            text: Optional transcript (if None, uses Whisper)
            language: Language code for Whisper

        Returns:
            List of timed VisemeFrame objects
        """
        if self.transcriber is None:
            raise ImportError("Audio processing not available. Install: pip install openai-whisper")

        # Transcribe audio with word-level timing
        transcription = self.transcriber.transcribe(audio_path, language=language)

        # Override with provided text if given
        if text is not None:
            transcription.text = text

        # Align visemes with audio timing
        visemes = self.aligner.align_visemes_to_audio(transcription, self.text_converter)

        # Smooth timing
        visemes = self.aligner.smooth_viseme_timing(visemes)

        return visemes


class VisemeAnimationGenerator:
    """Generate animation keyframes from viseme sequence - SIMPLIFIED"""

    def __init__(self, fps: int = 25, smoothing: float = 0.1):
        """
        Args:
            fps: Target animation framerate
            smoothing: Interpolation smoothing (0=none, 1=full)
        """
        self.fps = fps
        self.smoothing = smoothing
        self.mapper = VisemeToLipMapper()

    def generate_keyframes(self, visemes: List[VisemeFrame]) -> List[float]:
        """
        Generate per-frame lip_open values from viseme sequence

        Returns:
            List of lip_open values (0.0 to 0.8) for each frame
        """
        if not visemes:
            return []

        # Calculate total duration and frame count
        total_duration = max(v.end_time for v in visemes)
        frame_count = int(total_duration * self.fps) + 1

        keyframes = []

        for frame_idx in range(frame_count):
            frame_time = frame_idx / self.fps

            # Find active visemes at this time
            active_visemes = [v for v in visemes
                            if v.start_time <= frame_time < v.end_time]

            if not active_visemes:
                # Use silence
                lip_open = 0.0
            else:
                # Use the viseme
                current_viseme = active_visemes[0]
                lip_open = self.mapper.get_lip_open(
                    current_viseme.viseme,
                    current_viseme.intensity
                )

                # Apply smoothing at viseme transitions
                if self.smoothing > 0 and frame_idx > 0:
                    prev_lip_open = keyframes[-1]
                    lip_open = self.mapper.interpolate_lip_open(
                        lip_open, prev_lip_open, self.smoothing
                    )

            keyframes.append(lip_open)

        return keyframes

    def export_to_json(self, keyframes: List[float], output_path: str):
        """Export animation to JSON format"""
        animation_data = {
            "fps": self.fps,
            "frame_count": len(keyframes),
            "keyframes": keyframes
        }

        with open(output_path, 'w') as f:
            json.dump(animation_data, f, indent=2)
