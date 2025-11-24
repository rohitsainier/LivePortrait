# coding: utf-8
"""
Audio processing with Whisper transcription and timing extraction
Place this file at: src/utils/audio_processor.py
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper not installed. Install with: pip install openai-whisper")

try:
    from pydub import AudioSegment
    import librosa
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Warning: Audio libraries not available. Install: pip install pydub librosa")


@dataclass
class WordTiming:
    """Word-level timing information from Whisper"""
    word: str
    start: float  # seconds
    end: float    # seconds
    probability: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class AudioTranscription:
    """Complete transcription with timing"""
    text: str
    language: str
    duration: float
    words: List[WordTiming]
    segments: List[Dict]  # Whisper segments


class WhisperTranscriber:
    """Transcribe audio with word-level timestamps using Whisper"""

    def __init__(self, model_size: str = "base", device: str = "cuda"):
        """
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (cuda, cpu)
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper not installed. Install: pip install openai-whisper")

        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model '{model_size}' on {self.device}...")
        self.model = whisper.load_model(model_size, device=self.device)
        print(f"✓ Whisper model loaded successfully")

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> AudioTranscription:
        """
        Transcribe audio file with word-level timing

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es', 'fr'). Auto-detect if None

        Returns:
            AudioTranscription with word-level timing
        """
        print(f"Transcribing audio: {audio_path}")

        # Transcribe with word timestamps
        result = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            verbose=False
        )

        # Extract word-level timing
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append(WordTiming(
                    word=word_info["word"].strip(),
                    start=word_info["start"],
                    end=word_info["end"],
                    probability=word_info.get("probability", 1.0)
                ))

        # Get audio duration
        duration = self._get_audio_duration(audio_path)

        transcription = AudioTranscription(
            text=result["text"].strip(),
            language=result["language"],
            duration=duration,
            words=words,
            segments=result["segments"]
        )

        print(f"✓ Transcribed: '{transcription.text}'")
        print(f"  Language: {transcription.language}")
        print(f"  Duration: {transcription.duration:.2f}s")
        print(f"  Words: {len(transcription.words)}")

        return transcription

    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds"""
        if AUDIO_LIBS_AVAILABLE:
            try:
                audio = AudioSegment.from_file(audio_path)
                return len(audio) / 1000.0
            except:
                pass

        # Fallback to librosa
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return len(y) / sr
        except:
            return 0.0

    def export_transcription(self, transcription: AudioTranscription, output_path: str):
        """Export transcription to JSON"""
        data = {
            "text": transcription.text,
            "language": transcription.language,
            "duration": transcription.duration,
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability
                }
                for w in transcription.words
            ]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Transcription saved to: {output_path}")


class AudioVisemeAligner:
    """Align visemes with audio using transcription timing"""

    def __init__(self):
        """Initialize aligner"""
        pass

    def align_visemes_to_audio(
        self,
        transcription: AudioTranscription,
        viseme_converter
    ) -> List:
        """
        Create visemes aligned with audio timing from transcription

        Args:
            transcription: AudioTranscription with word timing
            viseme_converter: TextToVisemeConverter instance

        Returns:
            List of VisemeFrame objects with accurate timing
        """
        from .viseme_mapper import VisemeFrame, VisemeType

        if not transcription.words:
            # No word timing, fall back to uniform distribution
            return viseme_converter.text_to_visemes(
                transcription.text,
                duration=transcription.duration
            )

        # Convert each word to phonemes/visemes with timing
        all_visemes = []

        for word_timing in transcription.words:
            word = word_timing.word
            word_start = word_timing.start
            word_duration = word_timing.duration

            # Convert word to visemes
            word_visemes = viseme_converter.text_to_visemes(
                word,
                duration=word_duration
            )

            # Adjust timing to absolute time
            for viseme in word_visemes:
                viseme.start_time += word_start
                all_visemes.append(viseme)

        # Add silence at the end if needed
        if all_visemes:
            last_end = all_visemes[-1].end_time
            if last_end < transcription.duration:
                all_visemes.append(VisemeFrame(
                    viseme=VisemeType.SILENCE,
                    start_time=last_end,
                    duration=transcription.duration - last_end
                ))

        # Sort by start time
        all_visemes.sort(key=lambda v: v.start_time)

        # Fill gaps with silence
        filled_visemes = []
        prev_end = 0.0

        for viseme in all_visemes:
            if viseme.start_time > prev_end + 0.01:  # Gap detected
                filled_visemes.append(VisemeFrame(
                    viseme=VisemeType.SILENCE,
                    start_time=prev_end,
                    duration=viseme.start_time - prev_end
                ))
            filled_visemes.append(viseme)
            prev_end = viseme.end_time

        return filled_visemes

    def smooth_viseme_timing(
        self,
        visemes: List,
        min_duration: float = 0.03,
        blend_duration: float = 0.02
    ) -> List:
        """
        Smooth viseme transitions and enforce minimum durations

        Args:
            visemes: List of VisemeFrame objects
            min_duration: Minimum viseme duration in seconds
            blend_duration: Duration for blending between visemes

        Returns:
            Smoothed viseme list
        """
        if not visemes:
            return []

        smoothed = []
        i = 0

        while i < len(visemes):
            current = visemes[i]

            # Merge very short visemes with neighbors
            if current.duration < min_duration and i < len(visemes) - 1:
                next_viseme = visemes[i + 1]
                # Extend next viseme to cover current
                next_viseme.start_time = current.start_time
                next_viseme.duration += current.duration
                i += 1
                continue

            smoothed.append(current)
            i += 1

        return smoothed


def extract_audio_info(audio_path: str) -> Dict:
    """Extract basic audio information"""
    if not AUDIO_LIBS_AVAILABLE:
        return {"duration": 0.0, "sample_rate": 0, "channels": 0}

    try:
        audio = AudioSegment.from_file(audio_path)
        return {
            "duration": len(audio) / 1000.0,
            "sample_rate": audio.frame_rate,
            "channels": audio.channels,
            "format": audio_path.split('.')[-1]
        }
    except Exception as e:
        print(f"Error extracting audio info: {e}")
        return {"duration": 0.0, "sample_rate": 0, "channels": 0}
