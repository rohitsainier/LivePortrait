# coding: utf-8

"""
Command-line interface for audio-driven lip-sync using LivePortrait
"""

import os
import tyro
import subprocess
import os.path as osp
from dataclasses import dataclass
from typing import Literal

from src.audio_lipsync_pipeline import AudioLipSyncPipeline
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig
from src.utils.rprint import rlog as log


@dataclass
class AudioLipSyncConfig:
    """Configuration for audio lip-sync"""

    # Input/Output
    source_video: str = 'assets/examples/source/s13.mp4'  # Source video with face
    audio: str = 'path/to/audio.wav'  # Audio file to lip-sync to
    output: str = 'animations/lipsync_output.mp4'  # Output video path

    # Audio processing
    whisper_model: Literal['tiny', 'base', 'small', 'medium', 'large'] = 'base'  # Whisper model size
    language: str = 'en'  # Audio language (en, zh, es, fr, de, etc.)

    # Viseme settings
    viseme_set: Literal['liveportrait', 'preston_blair', 'oculus'] = 'liveportrait'  # Viseme mapping set
    enable_coarticulation: bool = True  # Apply coarticulation effects
    enable_smoothing: bool = True  # Apply temporal smoothing

    # LivePortrait settings
    retargeting_source_scale: float = 2.3  # Crop scale for source video
    driving_smooth_observation_variance: float = 3e-6  # Motion smoothing strength
    flag_do_crop: bool = True  # Crop source video face

    # System
    device: str = None  # Device (cuda/mps/cpu, auto-detect if None)
    flag_force_cpu: bool = False  # Force CPU inference

     # Enhanced controls
    expression_multiplier: float = 1.3  # Scale factor for lip movements (0.5=subtle, 1.5=exaggerated)
    coarticulation_factor: float = 0.30  # Phoneme blending (0=none, 0.5=high)
    smoothing: bool = True  # Apply temporal smoothing

    enable_temporal_dynamics: bool = True  # Enable anticipation/overshoot
    lip_sync_strength: float = 1.0  # Global strength multiplier (0.5-2.0)


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def main():
    # Parse arguments
    tyro.extras.set_accent_color("bright_cyan")
    config = tyro.cli(AudioLipSyncConfig)

    # Check FFmpeg
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg before running this script."
        )

    # Validate inputs
    if not osp.exists(config.source_video):
        raise FileNotFoundError(f"Source video not found: {config.source_video}")
    if not osp.exists(config.audio):
        raise FileNotFoundError(f"Audio file not found: {config.audio}")

    # Initialize LivePortrait configs
    inference_cfg = InferenceConfig()
    if config.flag_force_cpu:
        inference_cfg.flag_force_cpu = True

    crop_cfg = CropConfig()
    crop_cfg.scale = config.retargeting_source_scale

    # Initialize enhanced pipeline
    from src.audio_lipsync_pipeline import AudioLipSyncPipeline

    pipeline = AudioLipSyncPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg,
        whisper_model=config.whisper_model,
        device=config.device
    )

    # Execute with enhanced controls
    output_video, output_concat = pipeline.execute_audio_lipsync(
        source_video_path=config.source_video,
        audio_path=config.audio,
        output_path=config.output,
        language=config.language,
        smoothing=config.smoothing,
        coarticulation_factor=config.coarticulation_factor,
        expression_multiplier=config.expression_multiplier,
        enable_temporal_dynamics=config.enable_temporal_dynamics  # NEW
    )

    log("✅ Lip-sync complete!")
    log(f"Output video: {output_video}")
    log(f"Comparison video: {output_concat}")


if __name__ == '__main__':
    main()
