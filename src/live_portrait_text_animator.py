# coding: utf-8
"""
SIMPLIFIED Integration module for LivePortrait text/audio-driven animation
Uses ONLY retarget_lip function with single lip_open parameter
NOW WITH JSON FRAME TRACKING!
"""

import numpy as np
import cv2
import torch
import os
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from .utils.viseme_mapper import (
    TextToVisemeConverter,
    AudioToVisemeConverter,
    VisemeAnimationGenerator,
    VisemeFrame,
    PHONEMIZER_AVAILABLE
)
from .utils.io import load_image_rgb
from .utils.video import images2video, add_audio_to_video
from .config.crop_config import CropConfig


class LivePortraitTextAnimator:
    """SIMPLIFIED: Integrates text-to-viseme with LivePortrait using ONLY retarget_lip"""

    def __init__(self, gradio_pipeline, crop_cfg: CropConfig = None, whisper_model: str = "base"):
        """
        Args:
            gradio_pipeline: Instance of GradioPipeline
            crop_cfg: CropConfig instance (if None, creates default)
            whisper_model: Whisper model size for audio transcription
        """
        self.pipeline = gradio_pipeline
        self.crop_cfg = crop_cfg if crop_cfg is not None else CropConfig()
        self.text_converter = TextToVisemeConverter()
        self.audio_converter = AudioToVisemeConverter(whisper_model=whisper_model)
        self.anim_generator = VisemeAnimationGenerator(fps=25, smoothing=0.15)

    def animate_from_text(self,
                         source_image_path: str,
                         text: str,
                         output_path: str,
                         words_per_minute: float = 150,
                         crop_scale: float = 2.5,
                         flag_do_crop: bool = True,
                         flag_stitching: bool = True) -> str:
        """Animate portrait speaking the given text"""

        print(f"Converting text to visemes: '{text}'")
        visemes = self.text_converter.text_to_visemes(text, words_per_minute=words_per_minute)

        print(f"Generated {len(visemes)} visemes")
        keyframes = self.anim_generator.generate_keyframes(visemes)

        print(f"Generated {len(keyframes)} animation frames (lip_open values)")

        # Pass visemes for JSON generation
        return self._apply_keyframes(
            source_image_path, keyframes, output_path,
            crop_scale, flag_do_crop, flag_stitching,
            visemes=visemes, input_text=text
        )

    def animate_from_audio(self,
                          source_image_path: str,
                          audio_path: str,
                          output_path: str,
                          text: Optional[str] = None,
                          language: Optional[str] = None,
                          crop_scale: float = 2.5,
                          flag_do_crop: bool = True,
                          flag_stitching: bool = True,
                          fps: int = 25) -> str:
        """
        Animate portrait synchronized with audio using Whisper transcription

        Args:
            source_image_path: Path to source portrait
            audio_path: Path to audio file
            output_path: Output video path
            text: Optional manual transcript (if None, uses Whisper)
            language: Language code for Whisper (e.g., 'en', 'es')
            crop_scale: Face crop scale
            flag_do_crop: Whether to crop face
            flag_stitching: Whether to use stitching
            fps: Output video framerate

        Returns:
            Path to generated video with audio
        """
        try:
            print("="*60)
            print("Audio-Driven Animation with Whisper")
            print("="*60)

            # Update FPS
            self.anim_generator.fps = fps

            # Transcribe and convert audio to visemes
            print(f"\n📝 Transcribing audio: {audio_path}")
            if text:
                print(f"   Using provided transcript: '{text}'")

            visemes = self.audio_converter.audio_to_visemes(
                audio_path=audio_path,
                text=text,
                language=language
            )

            print(f"\n✓ Generated {len(visemes)} visemes from audio")

            # Generate animation keyframes
            print(f"📊 Generating animation keyframes at {fps} FPS...")
            keyframes = self.anim_generator.generate_keyframes(visemes)
            print(f"✓ Generated {len(keyframes)} animation frames (lip_open values)")

            # Get transcription text
            transcription_text = text if text else getattr(
                self.audio_converter.transcriber.transcribe(audio_path) if self.audio_converter.transcriber else None,
                'text',
                'No transcription'
            )

            # Generate video without audio first
            video_no_audio = output_path.replace('.mp4', '_no_audio.mp4')
            print(f"\n🎬 Generating animation...")

            video_path = self._apply_keyframes(
                source_image_path,
                keyframes,
                video_no_audio,
                crop_scale,
                flag_do_crop,
                flag_stitching,
                visemes=visemes,
                input_text=transcription_text,
                audio_path=audio_path
            )

            # Add audio to video
            print(f"\n🔊 Adding audio to video...")
            add_audio_to_video(video_path, audio_path, output_path)

            # Clean up intermediate file
            if os.path.exists(video_no_audio):
                os.remove(video_no_audio)

            print(f"\n✅ Complete! Video with audio saved to: {output_path}")
            print("="*60)

            return output_path

        except Exception as e:
            print(f"\n❌ Error in audio animation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _apply_keyframes(self, source_path, keyframes, output_path,
                        crop_scale, flag_do_crop, flag_stitching,
                        visemes: Optional[List[VisemeFrame]] = None,
                        input_text: str = "",
                        audio_path: Optional[str] = None):
        """
        SIMPLIFIED: Apply animation keyframes using ONLY retarget_lip
        NOW WITH JSON FRAME TRACKING!

        Args:
            keyframes: List of lip_open values (0.0 to 0.8)
            visemes: Optional list of VisemeFrame objects for metadata
            input_text: Original input text
            audio_path: Optional audio file path
        """

        source_img = load_image_rgb(source_path)

        if flag_do_crop:
            # Update crop scale
            self.crop_cfg.scale = crop_scale

            crop_info = self.pipeline.cropper.crop_source_image(
                source_img,
                self.crop_cfg
            )
            if crop_info is None:
                raise Exception("No face detected in source image!")
            img_crop_256x256 = crop_info['img_crop_256x256']
            source_lmk = crop_info['lmk_crop']
        else:
            img_crop_256x256 = cv2.resize(source_img, (256, 256))
            source_lmk = self.pipeline.cropper.calc_lmk_from_cropped_image(source_img)

        # Prepare source
        I_s = self.pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = self.pipeline.live_portrait_wrapper.get_kp_info(I_s)
        x_s = self.pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)
        f_s = self.pipeline.live_portrait_wrapper.extract_feature_3d(I_s)

        output_frames = []

        # JSON tracking data
        frame_metadata = []
        animation_metadata = {
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "source_image": os.path.basename(source_path),
            "output_video": os.path.basename(output_path),
            "audio_file": os.path.basename(audio_path) if audio_path else None,
            "input_text": input_text,
            "configuration": {
                "fps": self.anim_generator.fps,
                "crop_scale": crop_scale,
                "flag_do_crop": flag_do_crop,
                "flag_stitching": flag_stitching,
                "smoothing": self.anim_generator.smoothing,
            },
            "statistics": {
                "total_frames": len(keyframes),
                "duration_seconds": len(keyframes) / self.anim_generator.fps,
                "total_visemes": len(visemes) if visemes else 0,
            },
            "frames": []
        }

        print(f"🎨 Rendering {len(keyframes)} frames using retarget_lip...")

        for frame_idx, lip_open in enumerate(keyframes):
            # Calculate frame time
            frame_time = frame_idx / self.anim_generator.fps

            # Find active viseme at this frame time
            active_viseme = None
            if visemes:
                for viseme in visemes:
                    if viseme.start_time <= frame_time < viseme.end_time:
                        active_viseme = viseme
                        break

            # Convert lip_open (0.0-0.8) to lip_close_ratio format
            # retarget_lip expects [[ratio]] where 0=closed, higher=more open
            lip_close_ratio = [[lip_open]]

            # Calculate combined lip ratio using LivePortrait's function
            combined_lip_ratio = self.pipeline.live_portrait_wrapper.calc_combined_lip_ratio(
                lip_close_ratio, source_lmk
            )

            # Use LivePortrait's retarget_lip function - THIS IS THE KEY!
            lip_delta = self.pipeline.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio)

            # Create new keypoints with lip delta
            x_d_new = x_s.clone()
            x_d_new += lip_delta

            # Apply stitching if enabled
            if flag_stitching and self.pipeline.live_portrait_wrapper.stitching_retargeting_module is not None:
                x_d_new = self.pipeline.live_portrait_wrapper.stitching(x_s, x_d_new)

            # Warp and decode
            out = self.pipeline.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_new)
            frame = self.pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
            output_frames.append(frame)

            # Collect frame metadata
            frame_data = {
                "frame_number": frame_idx,
                "timestamp": round(frame_time, 4),
                "lip_open": round(lip_open, 4),
                "viseme": {
                    "type": active_viseme.viseme.value if active_viseme else "none",
                    "phoneme": active_viseme.phoneme if active_viseme else "",
                    "intensity": round(active_viseme.intensity, 3) if active_viseme else 0.0,
                    "start_time": round(active_viseme.start_time, 4) if active_viseme else 0.0,
                    "end_time": round(active_viseme.end_time, 4) if active_viseme else 0.0,
                } if active_viseme else None,
                "processing": {
                    "stitching_applied": flag_stitching and self.pipeline.live_portrait_wrapper.stitching_retargeting_module is not None,
                    "retarget_lip_used": True,
                }
            }

            animation_metadata["frames"].append(frame_data)

            if (frame_idx + 1) % 50 == 0 or frame_idx == len(keyframes) - 1:
                viseme_info = f", viseme={active_viseme.viseme.value}" if active_viseme else ""
                print(f"   Rendered {frame_idx + 1}/{len(keyframes)} frames (lip_open={lip_open:.3f}{viseme_info})")

        # Save video
        images2video(output_frames, wfp=output_path, fps=self.anim_generator.fps)
        print(f"✓ Animation saved to: {output_path}")

        # Save JSON metadata
        json_path = output_path.replace('.mp4', '_metadata.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(animation_metadata, f, indent=2, ensure_ascii=False)
        print(f"✓ Frame metadata saved to: {json_path}")

        # Also save a simplified viseme timeline
        if visemes:
            timeline_path = output_path.replace('.mp4', '_timeline.json')
            timeline_data = {
                "fps": self.anim_generator.fps,
                "duration": len(keyframes) / self.anim_generator.fps,
                "viseme_timeline": [
                    {
                        "viseme": v.viseme.value,
                        "phoneme": v.phoneme,
                        "start": round(v.start_time, 4),
                        "end": round(v.end_time, 4),
                        "duration": round(v.duration, 4),
                        "intensity": round(v.intensity, 3)
                    }
                    for v in visemes
                ]
            }
            with open(timeline_path, 'w', encoding='utf-8') as f:
                json.dump(timeline_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Viseme timeline saved to: {timeline_path}")

        return output_path
