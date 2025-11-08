# coding: utf-8

"""
Video-based Lip Parameter Analyzer
Extracts phoneme timeline from audio + tracks lip movements from video
Generates data-driven viseme mappings
"""

import os
import cv2
import json
import torch
import whisper
import librosa
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path
import warnings

# Import from your existing code
try:
    from g2p_en import G2p
    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    warnings.warn("g2p_en not installed. Install with: pip install g2p-en")


@dataclass
class LipParameters:
    """Multi-dimensional lip parameters"""
    jaw_open: float = 0.0
    lip_width: float = 0.5
    lip_protrusion: float = 0.0
    upper_lip_raise: float = 0.0
    lower_lip_lower: float = 0.0
    corner_pull_horizontal: float = 0.0
    corner_pull_back: float = 0.0
    lip_tightness: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class VideoLipTracker:
    """Extract lip parameters from video frames using MediaPipe"""

    # CORRECTED MediaPipe Face Mesh landmark indices
    # See: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
    UPPER_LIP_TOP = 13      # Top of upper lip (vermilion border)
    LOWER_LIP_BOTTOM = 14   # Bottom of lower lip (vermilion border)
    UPPER_LIP_INNER = 12    # Inner upper lip
    LOWER_LIP_INNER = 15    # Inner lower lip
    LIP_LEFT_CORNER = 61    # Left mouth corner
    LIP_RIGHT_CORNER = 291  # Right mouth corner
    NOSE_TIP = 1            # Nose tip
    CHIN = 152              # Chin point

    # Additional landmarks for better tracking
    UPPER_LIP_CENTER = 0    # Center of upper lip outer
    LOWER_LIP_CENTER = 17   # Center of lower lip outer

    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Face Mesh initialized")

    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks from a single frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]

        # Convert to numpy array
        h, w = frame.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.append([landmark.x * w, landmark.y * h, landmark.z * w])

        return np.array(landmarks)

    def calculate_distance(self, lmks: np.ndarray, idx1: int, idx2: int) -> float:
        """Calculate Euclidean distance between two landmarks"""
        return np.linalg.norm(lmks[idx1] - lmks[idx2])

    def calculate_lip_parameters(self, landmarks: np.ndarray) -> LipParameters:
        """Extract multi-dimensional lip parameters from landmarks"""
        params = LipParameters()

        # Face reference measurements
        face_height = self.calculate_distance(landmarks, self.NOSE_TIP, self.CHIN)

        if face_height == 0:
            return params  # Avoid division by zero

        # 1. JAW OPEN (FIXED normalization)
        # Use inner mouth opening for better accuracy
        mouth_height_inner = self.calculate_distance(landmarks, self.UPPER_LIP_INNER, self.LOWER_LIP_INNER)

        # FIXED: Use realistic normalization
        # Typical mouth opening when fully open is ~30-40% of nose-to-chin distance
        # We normalize to this range
        params.jaw_open = np.clip(mouth_height_inner / (face_height * 0.35), 0.0, 1.0)

        # 2. LIP WIDTH
        mouth_width = self.calculate_distance(landmarks, self.LIP_LEFT_CORNER, self.LIP_RIGHT_CORNER)
        # Typical neutral mouth width is ~45-50% of face height
        neutral_mouth_width = face_height * 0.48
        width_ratio = mouth_width / neutral_mouth_width
        params.lip_width = np.clip(width_ratio, 0.0, 1.0)

        # 3. LIP PROTRUSION (Z-depth)
        # Average Z position of lips vs nose
        lip_indices = [self.UPPER_LIP_TOP, self.LOWER_LIP_BOTTOM,
                       self.LIP_LEFT_CORNER, self.LIP_RIGHT_CORNER]
        avg_lip_z = np.mean([landmarks[idx][2] for idx in lip_indices])
        nose_z = landmarks[self.NOSE_TIP][2]

        # Positive when lips protrude forward
        protrusion = (nose_z - avg_lip_z) / (face_height * 0.15)
        params.lip_protrusion = np.clip(protrusion, 0.0, 1.0)

        # 4. UPPER LIP RAISE
        upper_lip_to_nose = landmarks[self.NOSE_TIP][1] - landmarks[self.UPPER_LIP_TOP][1]
        neutral_upper_distance = face_height * 0.18
        params.upper_lip_raise = np.clip(1.0 - (upper_lip_to_nose / neutral_upper_distance), 0.0, 1.0)

        # 5. LOWER LIP LOWER
        lower_lip_to_chin = landmarks[self.CHIN][1] - landmarks[self.LOWER_LIP_BOTTOM][1]
        neutral_lower_distance = face_height * 0.22
        params.lower_lip_lower = np.clip(1.0 - (lower_lip_to_chin / neutral_lower_distance), 0.0, 1.0)

        # 6. CORNER PULL HORIZONTAL (smile vs frown)
        left_corner_y = landmarks[self.LIP_LEFT_CORNER][1]
        right_corner_y = landmarks[self.LIP_RIGHT_CORNER][1]
        avg_corner_y = (left_corner_y + right_corner_y) / 2

        # Compare to midpoint of lips
        lip_center_y = (landmarks[self.UPPER_LIP_TOP][1] + landmarks[self.LOWER_LIP_BOTTOM][1]) / 2

        # Negative = corners pulled down (frown), Positive = corners pulled up (smile)
        corner_lift = (lip_center_y - avg_corner_y) / (face_height * 0.08)
        params.corner_pull_horizontal = np.clip(corner_lift, -1.0, 1.0)

        # 7. CORNER PULL BACK (spread)
        # How much wider than neutral
        params.corner_pull_back = np.clip((width_ratio - 1.0), 0.0, 1.0)

        # 8. LIP TIGHTNESS
        # Inversely related to mouth opening
        params.lip_tightness = np.clip(1.0 - (params.jaw_open * 0.5), 0.0, 1.0)

        return params


class AudioPhonemeExtractor:
    """Extract phoneme timeline from audio using Whisper + G2P"""

    def __init__(self, whisper_model: str = 'base', device: str = None):
        """Initialize Whisper and G2P"""
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        # Load Whisper on CPU if MPS (compatibility)
        whisper_device = 'cpu' if self.device == 'mps' else self.device

        print(f"Loading Whisper model: {whisper_model} on {whisper_device}")
        self.whisper_model = whisper.load_model(whisper_model, device=whisper_device)
        print("✅ Whisper model loaded")

        # Initialize G2P
        if G2P_AVAILABLE:
            self.g2p = G2p()
            print("✅ G2P (Grapheme-to-Phoneme) engine initialized")
        else:
            raise ImportError("g2p_en is required. Install with: pip install g2p-en")

    def transcribe_audio(self, audio_path: str, language: str = 'en') -> Dict:
        """Transcribe audio using Whisper"""
        print(f"📝 Transcribing audio: {audio_path}")

        use_fp16 = (self.whisper_model.device.type == 'cuda')

        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            fp16=use_fp16
        )

        print(f"✅ Transcription: {result['text']}")
        return result

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phonemes using G2P"""
        phonemes = self.g2p(text)
        # Remove punctuation
        phonemes = [p for p in phonemes if p not in [' ', ',', '.', '!', '?', '-', "'"]]
        return phonemes

    def extract_phoneme_timeline(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> List[Tuple[str, float, float]]:
        """
        Extract phoneme timeline with timestamps
        Returns: List of (phoneme, start_time, end_time)
        """
        # Transcribe
        transcription = self.transcribe_audio(audio_path, language)

        # Extract words with timestamps
        words_with_timestamps = []
        for segment in transcription['segments']:
            if 'words' in segment:
                words_with_timestamps.extend(segment['words'])

        if len(words_with_timestamps) == 0:
            # Fallback: use segment-level timestamps
            for segment in transcription['segments']:
                words_with_timestamps.append({
                    'word': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                })

        # Convert words to phonemes
        import re
        phoneme_timeline = []

        for word_info in words_with_timestamps:
            word_clean = re.sub(r'[^\w\s]', '', word_info['word'].strip())

            if not word_clean:
                continue

            word_start = word_info['start']
            word_end = word_info['end']
            word_duration = word_end - word_start

            # Convert word to phonemes
            phonemes = self.text_to_phonemes(word_clean)

            if len(phonemes) == 0:
                phoneme_timeline.append(('SIL', word_start, word_end))
                continue

            # Distribute phonemes evenly across word duration
            phoneme_duration = word_duration / len(phonemes)

            for i, phoneme in enumerate(phonemes):
                phoneme_start = word_start + i * phoneme_duration
                phoneme_end = phoneme_start + phoneme_duration
                phoneme_clean = phoneme.upper().replace('0', '').replace('1', '').replace('2', '')
                phoneme_timeline.append((phoneme_clean, phoneme_start, phoneme_end))

        print(f"✅ Extracted {len(phoneme_timeline)} phonemes from audio")
        return phoneme_timeline


class VideoLipAnalyzer:
    """Main class to analyze video and generate phoneme-to-lip mappings"""

    def __init__(self, whisper_model: str = 'base'):
        """Initialize analyzer"""
        self.lip_tracker = VideoLipTracker()
        self.phoneme_extractor = AudioPhonemeExtractor(whisper_model=whisper_model)

    def extract_audio_from_video(self, video_path: str, output_audio_path: str = None) -> str:
        """Extract audio from video file"""
        if output_audio_path is None:
            output_audio_path = video_path.replace('.mp4', '_audio.wav').replace('.avi', '_audio.wav')

        print(f"🎵 Extracting audio from video...")

        import subprocess
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_audio_path
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        print(f"✅ Audio extracted to: {output_audio_path}")

        return output_audio_path

    def analyze_video(
        self,
        video_path: str,
        language: str = 'en',
        output_json: str = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Main analysis pipeline:
        1. Extract audio from video
        2. Get phoneme timeline from audio
        3. Track lip movements from video frames
        4. Align and aggregate phoneme → lip parameters
        """
        print("=" * 80)
        print("🎬 Starting Video Lip Analysis")
        print("=" * 80)

        # Step 1: Extract audio
        audio_path = self.extract_audio_from_video(video_path)

        # Step 2: Get phoneme timeline
        phoneme_timeline = self.phoneme_extractor.extract_phoneme_timeline(audio_path, language)

        # Step 3: Get video properties
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        print(f"📹 Video: {total_frames} frames at {fps:.2f} fps ({duration:.2f}s)")

        # Step 4: Track lip parameters for each frame
        print(f"🔍 Tracking lip movements from video...")

        frame_lip_params = []
        frame_idx = 0

        from rich.progress import track

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.lip_tracker.extract_landmarks(frame)

            if landmarks is not None:
                params = self.lip_tracker.calculate_lip_parameters(landmarks)
            else:
                # No face detected, use neutral
                params = LipParameters()

            frame_lip_params.append(params)
            frame_idx += 1

            # Progress indicator
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx}/{total_frames} frames", end='\r')

        cap.release()
        print(f"\n✅ Tracked {len(frame_lip_params)} frames")

        # Step 5: Align phonemes with frame lip parameters
        print(f"🔗 Aligning phonemes with lip movements...")

        phoneme_to_params = defaultdict(list)

        for phoneme, start_time, end_time in phoneme_timeline:
            # Convert time to frame indices
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            # Clamp to valid range
            start_frame = max(0, min(start_frame, len(frame_lip_params) - 1))
            end_frame = max(0, min(end_frame, len(frame_lip_params)))

            # Collect lip parameters for this phoneme's duration
            for frame_idx in range(start_frame, end_frame):
                if frame_idx < len(frame_lip_params):
                    phoneme_to_params[phoneme].append(frame_lip_params[frame_idx].to_dict())

        # Step 6: Aggregate parameters per phoneme (average)
        print(f"📊 Aggregating parameters for {len(phoneme_to_params)} unique phonemes...")

        aggregated_phonemes = {}

        for phoneme, param_list in phoneme_to_params.items():
            if len(param_list) == 0:
                continue

            # Calculate average for each parameter
            avg_params = {}
            for key in param_list[0].keys():
                values = [p[key] for p in param_list]
                avg_params[key] = float(np.mean(values))

            aggregated_phonemes[phoneme] = avg_params
            print(f"  {phoneme}: {len(param_list)} samples → jaw_open={avg_params['jaw_open']:.3f}")

        # Step 7: Save to JSON
        if output_json is None:
            video_name = Path(video_path).stem
            output_json = f"lip_analysis_{video_name}.json"

        with open(output_json, 'w') as f:
            json.dump(aggregated_phonemes, f, indent=2)

        print("=" * 80)
        print(f"✅ Analysis complete!")
        print(f"📁 Saved to: {output_json}")
        print(f"📊 Extracted {len(aggregated_phonemes)} phoneme mappings")
        print("=" * 80)

        # Cleanup temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return aggregated_phonemes


def main():
    """Command-line interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze video to extract phoneme-to-lip parameter mappings"
    )
    parser.add_argument(
        'video',
        type=str,
        help='Path to input video file'
    )
    parser.add_argument(
        '-l', '--language',
        type=str,
        default='en',
        help='Audio language (en, zh, es, fr, de, etc.)'
    )
    parser.add_argument(
        '-w', '--whisper-model',
        type=str,
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output JSON file path'
    )

    args = parser.parse_args()

    # Check if video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        return

    # Run analysis
    analyzer = VideoLipAnalyzer(whisper_model=args.whisper_model)
    result = analyzer.analyze_video(
        video_path=args.video,
        language=args.language,
        output_json=args.output
    )

    print("\n💡 Next steps:")
    print(f"   1. Run: python analyze_lip_tracking.py {args.output or 'lip_analysis_*.json'}")
    print(f"   2. Review the comparison with current viseme mappings")
    print(f"   3. Copy values from updated_visemes.py to your code")


if __name__ == '__main__':
    main()
