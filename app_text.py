# coding: utf-8
"""
SIMPLIFIED Gradio app - Uses ONLY retarget_lip with single lip_open slider
"""

import os
import gradio as gr
import numpy as np
import cv2
import torch
from src.gradio_pipeline import GradioPipeline
from src.live_portrait_text_animator import LivePortraitTextAnimator
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.utils.io import load_image_rgb

# Initialize configurations
args = ArgumentConfig()
inference_cfg = InferenceConfig()
crop_cfg = CropConfig()

# Initialize pipeline
pipeline = GradioPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg, args=args)

# Initialize text animator with Whisper support
text_animator = LivePortraitTextAnimator(pipeline, crop_cfg=crop_cfg, whisper_model="base")

# Global variables for manual control
manual_control_x_s = None
manual_control_f_s = None
manual_control_source_lmk = None
manual_control_crop_info = None

def animate_from_text(source_image, text, wpm, crop_scale):
    """Generate animation from text"""
    try:
        if not source_image:
            return None, "❌ Please upload a source image"

        if not text or text.strip() == "":
            return None, "❌ Please enter text"

        output_path = "animations/text_output.mp4"
        os.makedirs("animations", exist_ok=True)

        result = text_animator.animate_from_text(
            source_image_path=source_image,
            text=text,
            output_path=output_path,
            words_per_minute=wpm,
            crop_scale=crop_scale,
            flag_do_crop=True,
            flag_stitching=True
        )

        return result, f"✅ Animation generated successfully!\n📁 Saved to: {result}"
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def animate_from_audio(source_image, audio_file, manual_text, language, crop_scale, fps):
    """Generate animation from audio with Whisper transcription"""
    try:
        if not source_image:
            return None, None, "❌ Please upload a source image"

        if not audio_file:
            return None, None, "❌ Please upload an audio file"

        output_path = "animations/audio_output.mp4"
        transcript_path = "animations/audio_transcript.txt"
        os.makedirs("animations", exist_ok=True)

        text_for_alignment = manual_text if manual_text.strip() else None

        result = text_animator.animate_from_audio(
            source_image_path=source_image,
            audio_path=audio_file,
            output_path=output_path,
            text=text_for_alignment,
            language=language if language != "auto" else None,
            crop_scale=crop_scale,
            flag_do_crop=True,
            flag_stitching=True,
            fps=fps
        )

        # Get transcription for display
        if hasattr(text_animator.audio_converter, 'transcriber') and text_animator.audio_converter.transcriber:
            try:
                transcription = text_animator.audio_converter.transcriber.transcribe(audio_file)
                transcript_text = f"Detected Language: {transcription.language}\n\n"
                transcript_text += f"Transcription:\n{transcription.text}\n\n"
                transcript_text += f"Duration: {transcription.duration:.2f}s\n"
                transcript_text += f"Words: {len(transcription.words)}"

                with open(transcript_path, 'w', encoding='utf-8') as f:
                    f.write(transcription.text)
            except:
                transcript_text = "Transcription not available"
        else:
            transcript_text = "Using manual text" if text_for_alignment else "Whisper not available"

        status = f"✅ Audio animation generated successfully!\n📁 Video: {result}\n📝 Transcript: {transcript_path}"

        return result, transcript_text, status

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, None, error_msg

def init_manual_control(source_image, crop_scale):
    """Initialize source image for manual control"""
    global manual_control_x_s, manual_control_f_s, manual_control_source_lmk, manual_control_crop_info

    try:
        if not source_image:
            return None, "❌ Please upload a source image"

        # Load and process source image
        source_img = load_image_rgb(source_image)

        # Update crop config
        text_animator.crop_cfg.scale = crop_scale

        # Crop source image
        crop_info = pipeline.cropper.crop_source_image(source_img, text_animator.crop_cfg)
        if crop_info is None:
            return None, "❌ No face detected in image"

        img_crop_256x256 = crop_info['img_crop_256x256']
        source_lmk = crop_info['lmk_crop']

        # Prepare source
        I_s = pipeline.live_portrait_wrapper.prepare_source(img_crop_256x256)
        x_s_info = pipeline.live_portrait_wrapper.get_kp_info(I_s)
        x_s = pipeline.live_portrait_wrapper.transform_keypoint(x_s_info)
        f_s = pipeline.live_portrait_wrapper.extract_feature_3d(I_s)

        # Store globally
        manual_control_x_s = x_s
        manual_control_f_s = f_s
        manual_control_source_lmk = source_lmk
        manual_control_crop_info = crop_info

        return img_crop_256x256, "✅ Image loaded! Adjust lip_open slider to see changes in real-time."

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def apply_manual_lip_control(lip_open, flag_stitching):
    """SIMPLIFIED: Apply ONLY lip_open using retarget_lip"""
    global manual_control_x_s, manual_control_f_s, manual_control_source_lmk

    try:
        if manual_control_x_s is None or manual_control_f_s is None:
            return None, "*Load an image first to see parameters*"

        # Convert lip_open to lip_close_ratio format
        lip_close_ratio = [[lip_open]]

        # Use LivePortrait's calc_combined_lip_ratio and retarget_lip
        combined_lip_ratio = pipeline.live_portrait_wrapper.calc_combined_lip_ratio(
            lip_close_ratio, manual_control_source_lmk
        )

        # THIS IS THE KEY: Use retarget_lip function!
        lip_delta = pipeline.live_portrait_wrapper.retarget_lip(
            manual_control_x_s,
            combined_lip_ratio
        )

        # Apply delta to keypoints
        x_d_new = manual_control_x_s.clone()
        x_d_new += lip_delta

        # Apply stitching for smoothing
        if flag_stitching and pipeline.live_portrait_wrapper.stitching_retargeting_module is not None:
            x_d_new = pipeline.live_portrait_wrapper.stitching(manual_control_x_s, x_d_new)

        # Generate image
        out = pipeline.live_portrait_wrapper.warp_decode(manual_control_f_s, manual_control_x_s, x_d_new)
        frame = pipeline.live_portrait_wrapper.parse_output(out['out'])[0]

        # Create parameter summary
        params_text = f"""
### 📊 Current Parameters (SIMPLIFIED)

**Using ONLY LivePortrait's `retarget_lip` function:**

- 👄 **Lip Open**: `{lip_open:.3f}` {'✅ ACTIVE' if lip_open > 0.01 else '⚪ closed'}
  - Range: 0.0 (closed) to 0.8 (wide open)
  - Internal processing by `retarget_lip()`

**Settings:**
- Stitching: `{'✅ ON (smoothing applied)' if flag_stitching else '❌ OFF (raw output)'}`

**💡 How it works:**
1. `lip_open` value → `calc_combined_lip_ratio()`
2. Combined ratio → `retarget_lip()` (handles ALL lip deformations)
3. Automatic lip movement generation!

**✨ All complex lip movements (width, rounding, teeth, etc.) are handled internally by `retarget_lip`!**
        """

        return frame, params_text

    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

def reset_lip_slider():
    """Reset lip_open slider to default"""
    return 0.0, True

def load_lip_preset(preset_name):
    """Load predefined lip_open presets for common visemes"""
    presets = {
        "Neutral/Silence": (0.0, True),
        "M/P/B (Lips Closed)": (0.0, True),
        "F/V (Teeth on Lip)": (0.15, True),
        "TH (Tongue/Teeth)": (0.20, True),
        "W/R (Pursed/Round)": (0.20, True),
        "L/T/D (Slight Open)": (0.25, True),
        "K/G (Medium Open)": (0.30, True),
        "I/IH (Small 'sit')": (0.20, True),
        "E/EH (Medium 'bed')": (0.35, True),
        "U/OO (Round 'boot')": (0.30, True),
        "O (Large Round 'go')": (0.50, True),
        "AI (Wide 'bite')": (0.50, True),
        "AA (Very Wide 'father')": (0.60, True),
        "AW (Wide Round 'how')": (0.60, True),
    }
    return presets.get(preset_name, presets["Neutral/Silence"])

# Build Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="LivePortrait SIMPLIFIED - retarget_lip Only") as demo:
    gr.Markdown("""
    # 🎤 LivePortrait: SIMPLIFIED Text, Audio & Lip Control

    **Uses ONLY `retarget_lip` function with single `lip_open` parameter!**
    """)

    with gr.Tabs() as tabs:
        # Tab 1: Text-Driven Animation
        with gr.Tab("📝 Text-Driven Animation"):
            gr.Markdown("### Convert text to realistic lip-synced animation using `retarget_lip`")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Source Portrait")
                    text_source_image = gr.Image(type="filepath", label="Upload Portrait Image")
                    gr.Examples(
                        examples=[
                            ["assets/examples/source/s9.jpg"],
                            ["assets/examples/source/s6.jpg"],
                            ["assets/examples/source/s10.jpg"],
                            ["assets/examples/source/s5.jpg"],
                            ["assets/examples/source/s7.jpg"],
                        ],
                        inputs=[text_source_image],
                        label="Example Portraits"
                    )

                with gr.Column():
                    gr.Markdown("#### Text Input")
                    text_input = gr.Textbox(
                        label="Text to Speak",
                        placeholder="Enter the text you want the portrait to speak...",
                        lines=8,
                        value="Hello! Welcome to LivePortrait. This technology can make any portrait speak naturally."
                    )

                    with gr.Row():
                        text_wpm = gr.Slider(
                            minimum=50,
                            maximum=250,
                            value=150,
                            step=10,
                            label="Speaking Speed (Words Per Minute)"
                        )
                        text_crop_scale = gr.Slider(
                            minimum=1.8,
                            maximum=3.2,
                            value=2.5,
                            step=0.1,
                            label="Face Crop Scale"
                        )

                    text_generate_btn = gr.Button("🚀 Generate Text Animation", variant="primary", size="lg")

            with gr.Row():
                text_output_video = gr.Video(label="Generated Animation")
                text_status = gr.Textbox(label="Status", lines=4)

            text_generate_btn.click(
                fn=animate_from_text,
                inputs=[text_source_image, text_input, text_wpm, text_crop_scale],
                outputs=[text_output_video, text_status]
            )

        # Tab 2: Audio-Driven Animation
        with gr.Tab("🎵 Audio-Driven Animation (Whisper)"):
            gr.Markdown("### Upload audio and automatically generate synchronized animation using `retarget_lip`")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Source Portrait")
                    audio_source_image = gr.Image(type="filepath", label="Upload Portrait Image")
                    gr.Examples(
                        examples=[
                            ["assets/examples/source/s9.jpg"],
                            ["assets/examples/source/s6.jpg"],
                            ["assets/examples/source/s10.jpg"],
                        ],
                        inputs=[audio_source_image]
                    )

                with gr.Column():
                    gr.Markdown("#### Audio Input")
                    audio_input = gr.Audio(
                        type="filepath",
                        label="Upload Audio File (MP3, WAV, etc.)"
                    )

                    manual_text_input = gr.Textbox(
                        label="Manual Transcript (Optional)",
                        placeholder="Leave empty for auto-transcription...",
                        lines=4
                    )

                    with gr.Row():
                        language_select = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"],
                            value="auto",
                            label="Language"
                        )
                        audio_fps = gr.Slider(
                            minimum=20,
                            maximum=30,
                            value=25,
                            step=1,
                            label="Output FPS"
                        )

                    audio_crop_scale = gr.Slider(
                        minimum=1.8,
                        maximum=3.2,
                        value=2.5,
                        step=0.1,
                        label="Face Crop Scale"
                    )

                    audio_generate_btn = gr.Button("🚀 Generate Audio Animation", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    audio_output_video = gr.Video(label="Generated Animation with Audio")
                with gr.Column():
                    transcript_output = gr.Textbox(
                        label="Whisper Transcription",
                        lines=8,
                        interactive=False
                    )

            audio_status = gr.Textbox(label="Status", lines=3)

            audio_generate_btn.click(
                fn=animate_from_audio,
                inputs=[
                    audio_source_image,
                    audio_input,
                    manual_text_input,
                    language_select,
                    audio_crop_scale,
                    audio_fps
                ],
                outputs=[audio_output_video, transcript_output, audio_status]
            )

        # Tab 3: SIMPLIFIED Manual Lip Control
        with gr.Tab("💋 Manual Lip Control (SIMPLIFIED)"):
            gr.Markdown("""
            ### 💋 SIMPLIFIED: Single `lip_open` Slider + `retarget_lip`
            Test the `retarget_lip` function with just ONE parameter!
            """)

            with gr.Row():
                # LEFT COLUMN - Controls
                with gr.Column(scale=3):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### 📤 Source Image")
                            manual_source_image = gr.Image(
                                type="filepath",
                                label="Upload Portrait",
                                height=220
                            )

                            manual_crop_scale = gr.Slider(
                                minimum=1.8,
                                maximum=3.2,
                                value=2.5,
                                step=0.1,
                                label="Crop Scale"
                            )

                            load_image_btn = gr.Button("📥 Load Image", variant="primary")

                            gr.Examples(
                                examples=[
                                    ["assets/examples/source/s9.jpg"],
                                    ["assets/examples/source/s6.jpg"],
                                    ["assets/examples/source/s10.jpg"],
                                    ["assets/examples/source/s5.jpg"],
                                ],
                                inputs=[manual_source_image],
                                label="Quick Load"
                            )

                        with gr.Column():
                            gr.Markdown("#### 📋 Viseme Presets")
                            preset_dropdown = gr.Dropdown(
                                choices=[
                                    "Neutral/Silence",
                                    "M/P/B (Lips Closed)",
                                    "F/V (Teeth on Lip)",
                                    "TH (Tongue/Teeth)",
                                    "W/R (Pursed/Round)",
                                    "L/T/D (Slight Open)",
                                    "K/G (Medium Open)",
                                    "I/IH (Small 'sit')",
                                    "E/EH (Medium 'bed')",
                                    "U/OO (Round 'boot')",
                                    "O (Large Round 'go')",
                                    "AI (Wide 'bite')",
                                    "AA (Very Wide 'father')",
                                    "AW (Wide Round 'how')",
                                ],
                                value="Neutral/Silence",
                                label="Load Preset Viseme"
                            )

                            with gr.Row():
                                load_preset_btn = gr.Button("📋 Load Preset", size="sm")
                                reset_btn = gr.Button("🔄 Reset", size="sm")

                            flag_stitching = gr.Checkbox(
                                value=True,
                                label="✨ Enable Stitching (Smooth Blending)"
                            )

                            manual_status = gr.Textbox(
                                label="Status",
                                lines=2,
                                placeholder="Upload an image to start..."
                            )

                    gr.Markdown("---")
                    gr.Markdown("#### 🎚️ SINGLE Parameter Control")

                    lip_open = gr.Slider(
                        minimum=0.0,
                        maximum=0.8,
                        value=0.0,
                        step=0.01,
                        label="👄 Lip Open (ONLY parameter needed!)",
                        info="0.0 = closed, 0.8 = wide open | All other lip movements handled by retarget_lip!"
                    )

                # RIGHT COLUMN - Output
                with gr.Column(scale=2):
                    gr.Markdown("#### 🖼️ Live Preview")
                    manual_output_image = gr.Image(
                        label="Result (Updates in Real-Time)",
                        type="numpy",
                        height=450
                    )

                    with gr.Accordion("📊 Parameter Info", open=True):
                        parameter_display = gr.Markdown(
                            value="*Load an image and adjust lip_open slider*"
                        )

                    with gr.Accordion("💡 How It Works", open=True):
                        gr.Markdown("""
                        **🔧 SIMPLIFIED Architecture:**

                        1. **Input**: Single `lip_open` value (0.0 to 0.8)
                        2. **Processing**:
                           ```python
                           lip_close_ratio = [[lip_open]]
                           combined_ratio = calc_combined_lip_ratio(lip_close_ratio, landmarks)
                           lip_delta = retarget_lip(keypoints, combined_ratio)
                           ```
                        3. **Output**: Complete lip deformation including:
                           - Vertical opening
                           - Horizontal stretch
                           - Lip rounding
                           - Teeth visibility
                           - Upper/lower lip positions
                           - Natural asymmetry

                        **✨ All complex lip movements are generated automatically by `retarget_lip`!**

                        **Common Values:**
                        - `0.0` - Closed (M, P, B)
                        - `0.15-0.25` - Small opening (F, V, TH, I)
                        - `0.30-0.40` - Medium (E, K, U, ER)
                        - `0.50-0.60` - Large (O, AA, AI, AW)
                        - `0.80` - Maximum opening
                        """)

            # Binding events
            load_image_btn.click(
                fn=init_manual_control,
                inputs=[manual_source_image, manual_crop_scale],
                outputs=[manual_output_image, manual_status]
            )

            lip_open.change(
                fn=apply_manual_lip_control,
                inputs=[lip_open, flag_stitching],
                outputs=[manual_output_image, parameter_display]
            )

            flag_stitching.change(
                fn=apply_manual_lip_control,
                inputs=[lip_open, flag_stitching],
                outputs=[manual_output_image, parameter_display]
            )

            reset_btn.click(
                fn=reset_lip_slider,
                inputs=[],
                outputs=[lip_open, flag_stitching]
            )

            load_preset_btn.click(
                fn=load_lip_preset,
                inputs=[preset_dropdown],
                outputs=[lip_open, flag_stitching]
            )

        # Tab 4: Info
        with gr.Tab("ℹ️ Information"):
            gr.Markdown("""
            ## ✨ SIMPLIFIED Architecture

            This version uses **ONLY LivePortrait's `retarget_lip` function** with a **single `lip_open` parameter**.

            ### Key Simplification

            **Before (Complex):**
            - 6+ parameters (lip_open, lip_width, mouth_round, lip_upper, lip_lower, teeth_visible)
            - Manual keypoint manipulation
            - Complex parameter tuning

            **After (SIMPLIFIED):**
            - **1 parameter**: `lip_open` (0.0 to 0.8)
            - **1 function**: `retarget_lip(keypoints, combined_ratio)`
            - Automatic handling of ALL lip deformations

            ### How It Works

            ```python
            # 1. Single input value
            lip_open = 0.6  # (0.0 to 0.8)

            # 2. Convert to LivePortrait format
            lip_close_ratio = [[lip_open]]
            combined_ratio = calc_combined_lip_ratio(lip_close_ratio, landmarks)

            # 3. Apply retarget_lip (handles EVERYTHING automatically)
            lip_delta = retarget_lip(keypoints, combined_ratio)

            # 4. Done! All lip movements are generated
            new_keypoints = keypoints + lip_delta
            ```

            ### What `retarget_lip` Handles Automatically

            - ✅ Vertical mouth opening
            - ✅ Horizontal lip stretch/compression
            - ✅ Lip rounding and pursing
            - ✅ Upper and lower lip positions
            - ✅ Teeth visibility
            - ✅ Natural asymmetry
            - ✅ Realistic deformations

            ### Viseme-to-lip_open Mapping

            | Viseme | lip_open | Description |
            |--------|----------|-------------|
            | Silence, M/P/B | 0.0 | Lips closed |
            | F/V | 0.15 | Teeth on lip |
            | TH, I, W | 0.20 | Slight opening |
            | L, DD, R | 0.25 | Small opening |
            | K, U, ER | 0.30 | Medium opening |
            | E, EI | 0.35 | Medium-large |
            | O, AI, OW | 0.45-0.50 | Large opening |
            | AA, AW | 0.60 | Very wide |
            | Maximum | 0.80 | Extreme opening |

            ### Benefits

            1. **Simplicity**: One slider instead of six
            2. **Accuracy**: Uses LivePortrait's trained retargeting
            3. **Natural**: Automatic realistic lip movements
            4. **Fast**: Less computation, fewer parameters
            5. **Debuggable**: Easy to understand and tune

            ### Requirements

            ```bash
            pip install openai-whisper pydub librosa phonemizer
            ```

            ### Tips

            - Start with viseme presets to learn values
            - `lip_open=0.0` for consonants like M, P, B
            - `lip_open=0.2-0.3` for most consonants
            - `lip_open=0.5-0.6` for open vowels like AA
            - Enable stitching for smoother blending
            - `retarget_lip` handles width, rounding, teeth automatically!
            """)

if __name__ == "__main__":
    demo.launch(
        server_port=8890,
        share=False,
        server_name="127.0.0.1"
    )
