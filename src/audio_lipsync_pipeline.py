# coding: utf-8

"""
Enhanced Audio-driven lip-sync pipeline with multi-dimensional lip control
"""

import os
import torch
import whisper
import numpy as np
import os.path as osp
from typing import List, Tuple, Optional, Dict
from rich.progress import track
import warnings

from .utils.phoneme_to_viseme import MultiDimensionalPhonemeMapper
from .gradio_pipeline import GradioPipeline
from .config.crop_config import CropConfig
from .config.inference_config import InferenceConfig
from .utils.rprint import rlog as log
from .utils.video import get_fps
from .utils.io import load_video


# Optional: G2P (Grapheme-to-Phoneme) support
try:
    from g2p_en import G2p

    # Auto-download NLTK data if missing
    import nltk
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger_eng')
    except LookupError:
        log("Downloading required NLTK data (one-time setup)...")
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
        nltk.download('cmudict', quiet=True)
        log("NLTK data downloaded successfully")

    G2P_AVAILABLE = True
except ImportError:
    G2P_AVAILABLE = False
    warnings.warn("g2p_en not installed. Install with: pip install g2p-en")


class AudioLipSyncPipeline:
    """Enhanced pipeline with multi-dimensional lip control"""

    def __init__(
    self,
    inference_cfg: InferenceConfig,
    crop_cfg: CropConfig,
    whisper_model: str = 'base',
    device: str = None
    ):
        """Initialize the enhanced audio lip-sync pipeline"""

        # Initialize LivePortrait pipeline
        self.gradio_pipeline = GradioPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            args=None
        )

        # Determine device for LivePortrait
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        log(f"Using device: {self.device}")

        # ⚠️ FIX: Force Whisper to CPU (MPS has compatibility issues)
        whisper_device = 'cpu' if self.device == 'mps' else self.device

        if self.device == 'mps':
            log("Note: Loading Whisper on CPU due to MPS compatibility issues")

        # Load Whisper model
        log(f"Loading Whisper model: {whisper_model}")
        self.whisper_model = whisper.load_model(whisper_model, device=whisper_device)
        log("Whisper model loaded successfully")

        # Initialize enhanced phoneme mapper
        self.phoneme_mapper = MultiDimensionalPhonemeMapper()
        log("Multi-dimensional viseme mapper initialized")

        # Initialize G2P if available
        if G2P_AVAILABLE:
            try:
                self.g2p = G2p()
                log("G2P (Grapheme-to-Phoneme) engine initialized")
            except Exception as e:
                log(f"Warning: Could not initialize G2P: {e}")
                self.g2p = None
        else:
            self.g2p = None

    def transcribe_audio(self, audio_path: str, language: str = 'en') -> Dict:
        """Transcribe audio using Whisper"""
        log(f"Transcribing audio: {audio_path}")

        # Determine if we can use FP16 (only on CUDA)
        use_fp16 = (self.whisper_model.device.type == 'cuda')

        result = self.whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            fp16=use_fp16  # Only use FP16 on CUDA
        )

        log(f"Transcription complete: {result['text']}")
        return result

    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phonemes"""
        if not self.g2p:
            raise RuntimeError("G2P engine not available")

        try:
            phonemes = self.g2p(text)
            phonemes = [p for p in phonemes if p not in [' ', ',', '.', '!', '?', '-', "'"]]
            return phonemes
        except Exception as e:
            log(f"G2P conversion failed for '{text}': {e}")
            return []

    def align_phonemes_to_words(
        self,
        words_with_timestamps: List[Dict],
        phonemes_per_word: List[List[str]]
    ) -> List[Tuple[str, float, float]]:
        """Align phonemes to word timestamps"""
        phoneme_timeline = []

        for word_info, phonemes in zip(words_with_timestamps, phonemes_per_word):
            word_start = word_info['start']
            word_end = word_info['end']
            word_duration = word_end - word_start

            if len(phonemes) == 0:
                phoneme_timeline.append(('SIL', word_start, word_end))
                continue

            phoneme_duration = word_duration / len(phonemes)

            for i, phoneme in enumerate(phonemes):
                phoneme_start = word_start + i * phoneme_duration
                phoneme_end = phoneme_start + phoneme_duration
                phoneme_clean = phoneme.upper().replace('0', '').replace('1', '').replace('2', '')
                phoneme_timeline.append((phoneme_clean, phoneme_start, phoneme_end))

        return phoneme_timeline

    def extract_phoneme_timeline_from_audio(
        self,
        audio_path: str,
        language: str = 'en'
    ) -> List[Tuple[str, float, float]]:
        """Extract phoneme timeline from audio"""

        # Transcribe
        transcription = self.transcribe_audio(audio_path, language)

        # Extract words
        words_with_timestamps = []
        for segment in transcription['segments']:
            if 'words' in segment:
                words_with_timestamps.extend(segment['words'])

        if len(words_with_timestamps) == 0:
            for segment in transcription['segments']:
                words_with_timestamps.append({
                    'word': segment['text'],
                    'start': segment['start'],
                    'end': segment['end']
                })

        # Convert to phonemes
        import re
        phonemes_per_word = []
        for word_info in words_with_timestamps:
            word_clean = re.sub(r'[^\w\s]', '', word_info['word'].strip())

            if not word_clean:
                phonemes_per_word.append([])
                continue

            try:
                phonemes = self.text_to_phonemes(word_clean)
                phonemes_per_word.append(phonemes)
            except Exception as e:
                log(f"Warning: Could not convert '{word_clean}' to phonemes: {e}")
                phonemes_per_word.append([])

        # Align
        phoneme_timeline = self.align_phonemes_to_words(words_with_timestamps, phonemes_per_word)
        log(f"Extracted {len(phoneme_timeline)} phonemes from audio")

        return phoneme_timeline

    def execute_audio_lipsync(
        self,
        source_video_path: str,
        audio_path: str,
        output_path: str = None,
        language: str = 'en',
        retargeting_source_scale: float = 2.3,
        driving_smooth_observation_variance: float = 3e-6,
        flag_do_crop: bool = True,
        smoothing: bool = True,
        coarticulation_factor: float = 0.25,
        expression_multiplier: float = 1.0,
        enable_temporal_dynamics: bool = True
    ) -> Tuple[str, str]:
        """
        Enhanced execution with multi-dimensional lip control

        Args:
            expression_multiplier: Scale factor for all lip movements (0.5=subtle, 1.5=exaggerated)
        """
        log("=" * 60)
        log("Starting Enhanced Audio-Driven Lip Sync Pipeline")
        log("=" * 60)

        # Extract phoneme timeline
        phoneme_timeline = self.extract_phoneme_timeline_from_audio(audio_path, language)

        if len(phoneme_timeline) == 0:
            raise RuntimeError("No phonemes extracted from audio")

        # Get video properties
        source_fps = int(get_fps(source_video_path))
        log(f"Source video FPS: {source_fps}")

        import librosa
        audio_duration = librosa.get_duration(path=audio_path)
        log(f"Audio duration: {audio_duration:.2f}s")

        # Convert to multi-dimensional viseme parameters
        viseme_timeline = self.phoneme_mapper.phoneme_sequence_to_viseme_timeline(phoneme_timeline)

        # Interpolate to per-frame parameters
        frame_viseme_params = self.phoneme_mapper.interpolate_viseme_params(
            viseme_timeline,
            fps=source_fps,
            total_duration=audio_duration,
            smoothing=smoothing,
            smoothing_window=5,
            coarticulation_factor=coarticulation_factor
        )

        if enable_temporal_dynamics:
            frame_viseme_params = self._apply_temporal_smoothing_with_overshoot(
                frame_viseme_params,
                fps=source_fps
            )
            log("Applied temporal dynamics (anticipation & overshoot)")

        log(f"Generated {len(frame_viseme_params)} frames of multi-dimensional lip parameters")

        # Apply expression multiplier
        if expression_multiplier != 1.0:
            for params in frame_viseme_params:
                for key in params:
                    if key != 'asymmetry_left_right':  # Don't scale asymmetry
                        params[key] *= expression_multiplier
                        params[key] = np.clip(params[key], 0.0, 1.0)

        # Apply to video
        output_video, output_concat = self._apply_multi_dimensional_lip_sync(
            source_video_path=source_video_path,
            frame_viseme_params=frame_viseme_params,
            output_path=output_path,
            retargeting_source_scale=retargeting_source_scale,
            flag_do_crop=flag_do_crop
        )

        # Add audio
        from .utils.video import add_audio_to_video
        if output_video and osp.exists(output_video):
            output_with_audio = output_video.replace('.mp4', '_with_audio.mp4')
            add_audio_to_video(output_video, audio_path, output_with_audio)
            os.replace(output_with_audio, output_video)
            log(f"Added audio to output video: {output_video}")

        log("=" * 60)
        log("Enhanced Audio Lip Sync Complete!")
        log(f"Output: {output_video}")
        log("=" * 60)

        return output_video, output_concat

    def _apply_multi_dimensional_lip_sync(
        self,
        source_video_path: str,
        frame_viseme_params: List[Dict[str, float]],
        output_path: str = None,
        retargeting_source_scale: float = 2.3,
        flag_do_crop: bool = True
    ) -> Tuple[str, str]:
        """Apply multi-dimensional lip control to video"""

        import cv2
        from .utils.helper import mkdir, basename, dct2device
        from .utils.crop import prepare_paste_back, paste_back
        from .utils.video import images2video, concat_frames
        from .utils.io import resize_to_limit, load_video

        device = self.gradio_pipeline.live_portrait_wrapper.device
        inference_cfg = self.gradio_pipeline.live_portrait_wrapper.inference_cfg

        # Load source video/image
        source_rgb_lst = load_video(source_video_path)
        source_rgb_lst = [resize_to_limit(img, inference_cfg.source_max_dim, inference_cfg.source_division) for img in source_rgb_lst]
        source_fps = int(get_fps(source_video_path))

        # ✅ FIX: Handle single image source for audio lip-sync
        is_source_image = len(source_rgb_lst) == 1
        target_frames = len(frame_viseme_params)

        if is_source_image:
            log(f"Source is a single image, duplicating for {target_frames} frames")
            # Duplicate the single image for all frames
            source_rgb_lst = source_rgb_lst * target_frames
            n_frames = target_frames
        else:
            n_frames = min(len(source_rgb_lst), target_frames)
            if len(source_rgb_lst) != target_frames:
                log(f"Warning: Source video has {len(source_rgb_lst)} frames, but audio needs {target_frames} frames. Using {n_frames}.")

        log(f"Processing {n_frames} frames with multi-dimensional lip control")

        # Crop source video
        if flag_do_crop:
            ret_s = self.gradio_pipeline.cropper.crop_source_video(source_rgb_lst, self.gradio_pipeline.cropper.crop_cfg)
            img_crop_256x256_lst = ret_s['frame_crop_lst']
            source_lmk_crop_lst = ret_s['lmk_crop_lst']
            source_M_c2o_lst = ret_s['M_c2o_lst']
            mask_ori_lst = [prepare_paste_back(inference_cfg.mask_crop, M, dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0])) for M in source_M_c2o_lst]
        else:
            source_lmk_crop_lst = self.gradio_pipeline.cropper.calc_lmks_from_cropped_video(source_rgb_lst)
            img_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in source_rgb_lst]
            source_M_c2o_lst, mask_ori_lst = None, None

        # ✅ IMPORTANT: Ensure we have the right number of frames
        actual_processed = min(len(img_crop_256x256_lst), n_frames)
        if actual_processed != n_frames:
            log(f"Warning: After cropping, only {actual_processed} frames available (expected {n_frames})")
            n_frames = actual_processed

        # Prepare features
        I_s_lst = self.gradio_pipeline.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst[:n_frames])
        c_s_eyes_lst, c_s_lip_lst = self.gradio_pipeline.live_portrait_wrapper.calc_ratio(source_lmk_crop_lst[:n_frames])
        source_template_dct = self.gradio_pipeline.make_motion_template(I_s_lst, c_s_eyes_lst, c_s_lip_lst, output_fps=source_fps)

        # Process each frame
        f_s_lst, x_s_lst, x_d_new_lst = [], [], []

        for i in track(range(n_frames), description='Applying multi-dimensional lip control...', total=n_frames):
            # ✅ FIX: Handle source image case (use frame 0 for all)
            source_idx = 0 if is_source_image else i

            x_s_info = source_template_dct['motion'][source_idx]
            x_s_info = dct2device(x_s_info, device)

            I_s = I_s_lst[source_idx]
            f_s = self.gradio_pipeline.live_portrait_wrapper.extract_feature_3d(I_s)
            x_s = x_s_info['x_s']

            # Get viseme parameters for this frame
            viseme_params = frame_viseme_params[i]

            # PASS 1: Primary jaw opening
            jaw_open = viseme_params['jaw_open']
            primary_lip_ratio = np.power(jaw_open, 0.85) * 1.0

            source_lmk = source_lmk_crop_lst[source_idx]
            combined_lip_ratio_tensor = self.gradio_pipeline.live_portrait_wrapper.calc_combined_lip_ratio(
                [[primary_lip_ratio]],
                source_lmk
            )
            lip_delta_primary = self.gradio_pipeline.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor)

            # PASS 2: Multi-dimensional fine control
            lip_delta_fine = self._calculate_fine_grained_lip_delta(x_s, viseme_params, device)

            # PASS 3: Width-specific adjustment
            lip_width = viseme_params['lip_width']
            if abs(lip_width - 0.5) > 0.1:
                width_bias_ratio = (lip_width - 0.5) * 0.25
                width_combined = self.gradio_pipeline.live_portrait_wrapper.calc_combined_lip_ratio(
                    [[width_bias_ratio]],
                    source_lmk
                )
                lip_delta_width = self.gradio_pipeline.live_portrait_wrapper.retarget_lip(x_s, width_combined) * 0.25
            else:
                lip_delta_width = torch.zeros_like(lip_delta_primary)

            # Combine all deltas
            x_d_new = x_s + lip_delta_primary + lip_delta_fine * 1.0 + lip_delta_width

            f_s_lst.append(f_s)
            x_s_lst.append(x_s)
            x_d_new_lst.append(x_d_new)

        # ✅ FIX: Ensure lists are not empty
        if len(f_s_lst) == 0 or len(x_d_new_lst) == 0:
            raise RuntimeError(f"No frames were processed! n_frames={n_frames}, source frames={len(source_rgb_lst)}")

        # Generate frames
        I_p_lst = []
        I_p_pstbk_lst = [] if flag_do_crop else None

        for i in track(range(n_frames), description='Generating lip-synced frames...', total=n_frames):
            source_idx = 0 if is_source_image else i

            x_s_i = x_s_lst[i].to(device)
            f_s_i = f_s_lst[i].to(device)
            x_d_new = x_d_new_lst[i]

            # Apply stitching
            x_d_new = self.gradio_pipeline.live_portrait_wrapper.stitching(x_s_i, x_d_new)

            # Warp and decode
            out = self.gradio_pipeline.live_portrait_wrapper.warp_decode(f_s_i, x_s_i, x_d_new)
            I_p_i = self.gradio_pipeline.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if flag_do_crop:
                I_p_pstbk = paste_back(I_p_i, source_M_c2o_lst[source_idx], source_rgb_lst[source_idx], mask_ori_lst[source_idx])
                I_p_pstbk_lst.append(I_p_pstbk)

        # ✅ FIX: Verify we have frames before saving
        if len(I_p_lst) == 0:
            raise RuntimeError("No output frames generated!")

        log(f"Generated {len(I_p_lst)} lip-synced frames")

        # Save outputs
        output_dir = osp.dirname(output_path) if output_path else 'animations'
        mkdir(output_dir)

        # Concatenated video
        frames_concatenated = concat_frames(
            driving_image_lst=None,
            source_image_lst=img_crop_256x256_lst[:n_frames],
            I_p_lst=I_p_lst
        )
        wfp_concat = output_path.replace('.mp4', '_concat.mp4') if output_path else osp.join(output_dir, f'{basename(source_video_path)}_lipsync_concat.mp4')
        images2video(frames_concatenated, wfp=wfp_concat, fps=source_fps)

        # Final video
        wfp = output_path if output_path else osp.join(output_dir, f'{basename(source_video_path)}_lipsync.mp4')
        if I_p_pstbk_lst:
            images2video(I_p_pstbk_lst, wfp=wfp, fps=source_fps)
        else:
            images2video(I_p_lst, wfp=wfp, fps=source_fps)

        return wfp, wfp_concat

    def _calculate_fine_grained_lip_delta(
        self,
        x_s: torch.Tensor,
        viseme_params: Dict[str, float],
        device: str
    ) -> torch.Tensor:
        """
        CORRECTED: Fine-grained lip adjustments using VALID LivePortrait indices

        LivePortrait uses 21 keypoints (indices 0-20), not 106!
        Based on the structure, lip-related keypoints are typically in the middle range.

        Key indices for lips (estimated from LivePortrait's 21-point system):
        - Index 6: Lower lip center
        - Index 12: Upper lip center
        - Index 8: Lip corner/width control
        - Index 14: Secondary lip control
        - Index 17, 19, 20: Additional expression points
        """
        delta = torch.zeros_like(x_s)

        # Extract parameters
        jaw_open = viseme_params['jaw_open']
        lip_width = viseme_params['lip_width']
        lip_protrusion = viseme_params['lip_protrusion']
        upper_lip_raise = viseme_params['upper_lip_raise']
        lower_lip_lower = viseme_params['lower_lip_lower']
        corner_pull_h = viseme_params['corner_pull_horizontal']
        corner_pull_back = viseme_params['corner_pull_back']

        # SAFE SCALING - much more conservative than before
        # Only manipulate known lip-related indices

        # 1. LIP WIDTH (horizontal stretch) - Index 8 and 14
        if abs(lip_width - 0.5) > 0.05:
            width_delta = (lip_width - 0.5) * 0.06  # Reduced from 0.12
            delta[0, 8, 0] += width_delta   # Lip corner horizontal
            delta[0, 14, 0] -= width_delta  # Opposite side

        # 2. LIP PROTRUSION (forward push) - Indices 6, 12
        if lip_protrusion > 0.05:
            prot_scale = lip_protrusion * 0.04  # Reduced from 0.08
            delta[0, 6, 2] += prot_scale   # Lower lip forward
            delta[0, 12, 2] += prot_scale  # Upper lip forward

        # 3. UPPER LIP RAISE - Index 12
        if upper_lip_raise > 0.03:
            raise_scale = upper_lip_raise * 0.05  # Reduced from 0.10
            delta[0, 12, 1] -= raise_scale  # Raise upper lip

        # 4. LOWER LIP LOWER - Index 6
        if lower_lip_lower > 0.03:
            lower_scale = lower_lip_lower * 0.05  # Reduced from 0.10
            delta[0, 6, 1] += lower_scale   # Lower the lower lip

        # 5. CORNER PULL (smile/frown) - Indices 8, 14
        if abs(corner_pull_h) > 0.05:
            corner_scale = corner_pull_h * 0.03  # Reduced from 0.06
            delta[0, 8, 1] -= corner_scale   # Corners up/down
            delta[0, 14, 1] -= corner_scale

        # 6. JAW-CORRELATED ADJUSTMENTS
        if jaw_open > 0.3:
            jaw_lip_coupling = (jaw_open - 0.3) * 0.08  # Reduced from 0.15
            delta[0, 6, 1] += jaw_lip_coupling  # Extra lower lip drop

        return delta

    def _apply_temporal_smoothing_with_overshoot(
    self,
    frame_viseme_params: List[Dict[str, float]],
    fps: int
) -> List[Dict[str, float]]:
        """
        Apply overshoot and anticipation for natural dynamics
        """
        import copy
        smoothed_params = copy.deepcopy(frame_viseme_params)

        for i in range(1, len(frame_viseme_params) - 1):
            prev_params = frame_viseme_params[i-1]
            curr_params = frame_viseme_params[i]
            next_params = frame_viseme_params[i+1]

            # Get overshoot factor
            overshoot = curr_params.get('overshoot_factor', 0.0)
            anticipation = next_params.get('anticipation_factor', 0.0)

            if overshoot > 0.05:
                # Add overshoot: briefly exceed target then settle
                for key in ['jaw_open', 'lip_protrusion']:
                    if key in curr_params:
                        target = curr_params[key]
                        prev_val = prev_params[key]
                        overshoot_amount = (target - prev_val) * overshoot * 0.15
                        smoothed_params[i][key] = min(1.0, target + overshoot_amount)

            if anticipation > 0.05:
                # Add anticipation: start moving toward next phoneme early
                for key in ['jaw_open', 'lip_width', 'lip_protrusion']:
                    if key in curr_params:
                        curr_val = curr_params[key]
                        next_val = next_params[key]
                        anticipation_amount = (next_val - curr_val) * anticipation * 0.2
                        smoothed_params[i][key] = np.clip(curr_val + anticipation_amount, 0.0, 1.0)

        return smoothed_params
