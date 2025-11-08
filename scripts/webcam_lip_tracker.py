# coding: utf-8

"""
Real-time Lip Parameter Tracker for Fine-tuning Viseme Mappings
Uses MediaPipe Face Mesh for detailed lip tracking
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime


@dataclass
class LipParameters:
    """Multi-dimensional lip parameters extracted from webcam"""
    jaw_open: float = 0.0
    lip_width: float = 0.5
    lip_protrusion: float = 0.0
    upper_lip_raise: float = 0.0
    lower_lip_lower: float = 0.0
    corner_pull_horizontal: float = 0.0
    corner_pull_back: float = 0.0
    lip_tightness: float = 0.5

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
        }


class LipTracker:
    """Real-time lip parameter extraction using MediaPipe Face Mesh"""

    # MediaPipe Face Mesh landmark indices for lips (468-point model)
    # Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    # Key lip landmarks
    UPPER_LIP_TOP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]  # Top of upper lip
    UPPER_LIP_BOTTOM = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]  # Bottom of upper lip
    LOWER_LIP_TOP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]  # Top of lower lip
    LOWER_LIP_BOTTOM = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]  # Bottom of lower lip

    LIP_LEFT_CORNER = 61   # Left mouth corner
    LIP_RIGHT_CORNER = 291  # Right mouth corner

    UPPER_LIP_CENTER = 0   # Cupid's bow
    LOWER_LIP_CENTER = 17  # Lower lip center

    # Face reference points
    NOSE_TIP = 1
    CHIN = 152

    def __init__(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Calibration values (neutral face)
        self.calibrated = False
        self.neutral_params = None

    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None

        # Get first face
        face_landmarks = results.multi_face_landmarks[0]

        # Convert to numpy array [x, y, z] for each landmark
        h, w = image.shape[:2]
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

        # 1. JAW OPEN (vertical mouth opening)
        # Distance between upper lip top and lower lip bottom
        upper_lip_top = landmarks[self.UPPER_LIP_CENTER]
        lower_lip_bottom = landmarks[self.LOWER_LIP_CENTER]
        mouth_height = self.calculate_distance(landmarks, self.UPPER_LIP_CENTER, self.LOWER_LIP_CENTER)

        # Normalize by face height (nose to chin)
        face_height = self.calculate_distance(landmarks, self.NOSE_TIP, self.CHIN)
        params.jaw_open = np.clip(mouth_height / (face_height * 0.15), 0.0, 1.0)

        # 2. LIP WIDTH (horizontal stretch)
        # Distance between lip corners
        mouth_width = self.calculate_distance(landmarks, self.LIP_LEFT_CORNER, self.LIP_RIGHT_CORNER)

        # Normalize by face width (approximate)
        neutral_mouth_width = face_height * 0.45  # Approximate neutral width
        width_ratio = mouth_width / neutral_mouth_width
        params.lip_width = np.clip(width_ratio, 0.0, 1.0)

        # 3. LIP PROTRUSION (forward push - using Z depth)
        # Average Z-depth of lip landmarks
        lip_indices = [self.UPPER_LIP_CENTER, self.LOWER_LIP_CENTER,
                       self.LIP_LEFT_CORNER, self.LIP_RIGHT_CORNER]
        avg_lip_z = np.mean([landmarks[idx][2] for idx in lip_indices])
        nose_z = landmarks[self.NOSE_TIP][2]

        protrusion = (nose_z - avg_lip_z) / (face_height * 0.1)
        params.lip_protrusion = np.clip(protrusion, 0.0, 1.0)

        # 4. UPPER LIP RAISE
        # Distance from upper lip to nose
        upper_lip_height = landmarks[self.NOSE_TIP][1] - landmarks[self.UPPER_LIP_CENTER][1]
        neutral_upper_height = face_height * 0.15
        params.upper_lip_raise = np.clip(1.0 - (upper_lip_height / neutral_upper_height), 0.0, 1.0)

        # 5. LOWER LIP LOWER
        # Distance from lower lip to chin
        lower_lip_distance = landmarks[self.CHIN][1] - landmarks[self.LOWER_LIP_CENTER][1]
        neutral_lower_distance = face_height * 0.2
        params.lower_lip_lower = np.clip(1.0 - (lower_lip_distance / neutral_lower_distance), 0.0, 1.0)

        # 6. CORNER PULL (smile/frown - vertical movement of corners)
        left_corner_y = landmarks[self.LIP_LEFT_CORNER][1]
        right_corner_y = landmarks[self.LIP_RIGHT_CORNER][1]
        avg_corner_y = (left_corner_y + right_corner_y) / 2

        # Compare to lip center height (smile = corners higher)
        lip_center_y = (landmarks[self.UPPER_LIP_CENTER][1] + landmarks[self.LOWER_LIP_CENTER][1]) / 2
        corner_lift = (lip_center_y - avg_corner_y) / (face_height * 0.05)
        params.corner_pull_horizontal = np.clip(corner_lift, -1.0, 1.0)

        # 7. CORNER PULL BACK (horizontal pull - smile width)
        # Measured as deviation from neutral width
        params.corner_pull_back = np.clip((width_ratio - 1.0), 0.0, 1.0)

        # 8. LIP TIGHTNESS (thickness of lips)
        # Distance between upper lip top and bottom
        upper_thickness = np.mean([
            self.calculate_distance(landmarks, self.UPPER_LIP_TOP[i], self.UPPER_LIP_BOTTOM[i])
            for i in range(min(len(self.UPPER_LIP_TOP), len(self.UPPER_LIP_BOTTOM)))
        ])
        lower_thickness = np.mean([
            self.calculate_distance(landmarks, self.LOWER_LIP_TOP[i], self.LOWER_LIP_BOTTOM[i])
            for i in range(min(len(self.LOWER_LIP_TOP), len(self.LOWER_LIP_BOTTOM)))
        ])

        avg_thickness = (upper_thickness + lower_thickness) / 2
        neutral_thickness = face_height * 0.02
        params.lip_tightness = np.clip(avg_thickness / neutral_thickness, 0.0, 1.0)

        return params

    def calibrate_neutral(self, landmarks: np.ndarray):
        """Calibrate neutral face position"""
        self.neutral_params = self.calculate_lip_parameters(landmarks)
        self.calibrated = True
        print("✅ Calibrated neutral face position")

    def draw_lip_landmarks(self, image: np.ndarray, landmarks: np.ndarray):
        """Draw lip landmarks on image for visualization"""
        # Draw lip outline
        lip_indices = (
            self.UPPER_LIP_TOP +
            self.LOWER_LIP_BOTTOM +
            [self.LIP_LEFT_CORNER, self.LIP_RIGHT_CORNER]
        )

        for idx in lip_indices:
            x, y = int(landmarks[idx][0]), int(landmarks[idx][1])
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        # Draw key points
        cv2.circle(image, (int(landmarks[self.UPPER_LIP_CENTER][0]),
                           int(landmarks[self.UPPER_LIP_CENTER][1])),
                   4, (255, 0, 0), -1)
        cv2.circle(image, (int(landmarks[self.LOWER_LIP_CENTER][0]),
                           int(landmarks[self.LOWER_LIP_CENTER][1])),
                   4, (0, 0, 255), -1)


class LipTrackerGUI:
    """GUI for real-time lip parameter visualization"""

    def __init__(self):
        self.tracker = LipTracker()
        self.recording = False
        self.recorded_params = []
        self.current_phoneme = "SIL"
        self.phoneme_recordings = {}

    def draw_parameter_bars(self, image: np.ndarray, params: LipParameters):
        """Draw parameter value bars on image"""
        bar_x = 20
        bar_y_start = 50
        bar_height = 20
        bar_spacing = 30
        bar_width = 200

        param_dict = params.to_dict()

        for i, (name, value) in enumerate(param_dict.items()):
            y = bar_y_start + i * bar_spacing

            # Background bar
            cv2.rectangle(image, (bar_x, y), (bar_x + bar_width, y + bar_height),
                         (50, 50, 50), -1)

            # Value bar (color based on value)
            fill_width = int(bar_width * value)
            color = (0, int(255 * value), int(255 * (1 - value)))
            cv2.rectangle(image, (bar_x, y), (bar_x + fill_width, y + bar_height),
                         color, -1)

            # Text label
            text = f"{name}: {value:.3f}"
            cv2.putText(image, text, (bar_x + bar_width + 10, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_instructions(self, image: np.ndarray):
        """Draw keyboard instructions"""
        instructions = [
            "Controls:",
            "C - Calibrate neutral face",
            "R - Start/Stop recording",
            "P - Set phoneme (type name)",
            "S - Save recordings to JSON",
            "Q - Quit"
        ]

        y = image.shape[0] - 150
        for i, text in enumerate(instructions):
            cv2.putText(image, text, (20, y + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Show current phoneme and recording status
        status_y = 30
        status_text = f"Phoneme: {self.current_phoneme}"
        cv2.putText(image, status_text, (image.shape[1] - 200, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if self.recording:
            rec_text = "RECORDING"
            cv2.putText(image, rec_text, (image.shape[1] - 200, status_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def save_recordings(self):
        """Save recorded phoneme parameters to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lip_tracking_data_{timestamp}.json"

        # Calculate average parameters for each phoneme
        output = {}
        for phoneme, param_list in self.phoneme_recordings.items():
            if len(param_list) > 0:
                avg_params = {}
                for key in param_list[0].keys():
                    avg_params[key] = float(np.mean([p[key] for p in param_list]))
                output[phoneme] = avg_params

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✅ Saved recordings to {filename}")
        print(f"   Recorded {len(output)} phonemes with {sum(len(v) for v in self.phoneme_recordings.values())} total samples")

    def run(self):
        """Run the webcam lip tracker"""
        cap = cv2.VideoCapture(0)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("🎥 Lip Parameter Tracker Started")
        print("=" * 60)
        print("Instructions:")
        print("  C - Calibrate neutral face (close mouth, neutral expression)")
        print("  R - Start/Stop recording current phoneme")
        print("  P - Set phoneme name (type and press Enter)")
        print("  S - Save all recordings to JSON")
        print("  Q - Quit")
        print("=" * 60)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            # Extract landmarks
            landmarks = self.tracker.extract_landmarks(frame)

            if landmarks is not None:
                # Calculate parameters
                params = self.tracker.calculate_lip_parameters(landmarks)

                # Draw visualization
                self.tracker.draw_lip_landmarks(frame, landmarks)
                self.draw_parameter_bars(frame, params)

                # Record if active
                if self.recording:
                    self.recorded_params.append(params.to_dict())

            # Draw UI
            self.draw_instructions(frame)

            # Show frame
            cv2.imshow('Lip Parameter Tracker', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                if landmarks is not None:
                    self.tracker.calibrate_neutral(landmarks)
            elif key == ord('r'):
                if self.recording:
                    # Stop recording and save
                    if len(self.recorded_params) > 0:
                        if self.current_phoneme not in self.phoneme_recordings:
                            self.phoneme_recordings[self.current_phoneme] = []
                        self.phoneme_recordings[self.current_phoneme].extend(self.recorded_params)
                        print(f"✅ Recorded {len(self.recorded_params)} frames for phoneme '{self.current_phoneme}'")
                    self.recorded_params = []
                    self.recording = False
                else:
                    # Start recording
                    self.recording = True
                    self.recorded_params = []
                    print(f"🔴 Recording phoneme '{self.current_phoneme}'...")
            elif key == ord('p'):
                # Set phoneme name
                print("Enter phoneme name (e.g., AA, IY, M, etc.): ", end='', flush=True)
                phoneme = input().strip().upper()
                if phoneme:
                    self.current_phoneme = phoneme
                    print(f"✅ Set current phoneme to '{self.current_phoneme}'")
            elif key == ord('s'):
                if len(self.phoneme_recordings) > 0:
                    self.save_recordings()
                else:
                    print("❌ No recordings to save")

        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point"""
    gui = LipTrackerGUI()
    gui.run()


if __name__ == '__main__':
    main()
