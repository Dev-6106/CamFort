# asl_oop_ui.py
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import time

# -------------------- Utilities for nice UI --------------------
def draw_transparent_rect(img: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int],
                          color: Tuple[int, int, int], alpha: float = 0.6, radius: int = 0) -> None:
    """Draw translucent rectangle (optionally with rounded corners)."""
    overlay = img.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text_with_shadow(img: np.ndarray, text: str, org: Tuple[int, int], font=cv2.FONT_HERSHEY_SIMPLEX,
                         scale: float = 1.0, color=(255, 255, 255), thickness: int = 2,
                         shadow_color=(0, 0, 0), shadow_offset=(2, 2), line_type=cv2.LINE_AA):
    """Draw text with a subtle shadow for better contrast."""
    x, y = org
    sx, sy = shadow_offset
    cv2.putText(img, text, (x + sx, y + sy), font, scale, shadow_color, thickness + 2, line_type)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, line_type)


def draw_hold_bar(img: np.ndarray, top_left: Tuple[int, int], size: Tuple[int, int],
                  progress: float, bg_color=(60, 60, 60), fg_color=(0, 200, 200)):
    """Draw a horizontal progress bar showing hold progress (0.0 - 1.0)."""
    x, y = top_left
    w, h = size
    # background
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    # foreground
    fill_w = int(w * np.clip(progress, 0.0, 1.0))
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), fg_color, -1)
    # border
    cv2.rectangle(img, (x, y), (x + w, y + h), (200, 200, 200), 1)


# -------------------- Detector base class --------------------
class Detector(ABC):
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        pass


# -------------------- ASL Detector (MediaPipe) --------------------
class ASLDetector(Detector):
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 landmark_history_size: int = 16):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_history = deque(maxlen=landmark_history_size)

    @staticmethod
    def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
        angle = math.degrees(radians)
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle

    @staticmethod
    def _get_distance(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
        return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2))

    def detect_asl_letter(self, hand_landmarks) -> Optional[str]:
        if not hand_landmarks:
            return None

        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

        # indices
        WRIST = 0
        THUMB_CMC = 1
        THUMB_MCP = 2
        THUMB_IP = 3
        THUMB_TIP = 4
        INDEX_MCP = 5
        INDEX_PIP = 6
        INDEX_DIP = 7
        INDEX_TIP = 8
        MIDDLE_MCP = 9
        MIDDLE_PIP = 10
        MIDDLE_DIP = 11
        MIDDLE_TIP = 12
        RING_MCP = 13
        RING_PIP = 14
        RING_DIP = 15
        RING_TIP = 16
        PINKY_MCP = 17
        PINKY_PIP = 18
        PINKY_DIP = 19
        PINKY_TIP = 20

        def is_finger_up(tip_idx: int, pip_idx: int) -> bool:
            return landmarks[tip_idx][1] < landmarks[pip_idx][1]

        def is_thumb_up() -> bool:
            return landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0] + 0.02

        def is_finger_bent(tip_idx: int, dip_idx: int, pip_idx: int) -> bool:
            angle = self.calculate_angle(landmarks[tip_idx][:2], landmarks[dip_idx][:2], landmarks[pip_idx][:2])
            return angle < 160

        fingers_up = [1 if is_thumb_up() else 0,
                      1 if is_finger_up(INDEX_TIP, INDEX_PIP) else 0,
                      1 if is_finger_up(MIDDLE_TIP, MIDDLE_PIP) else 0,
                      1 if is_finger_up(RING_TIP, RING_PIP) else 0,
                      1 if is_finger_up(PINKY_TIP, PINKY_PIP) else 0]

        thumb_index_distance = self._get_distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
        thumb_middle_distance = self._get_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_TIP])
        index_middle_distance = self._get_distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
        thumb_pinky_distance = self._get_distance(landmarks[THUMB_TIP], landmarks[PINKY_TIP])

        # A
        if fingers_up == [1, 0, 0, 0, 0] and thumb_index_distance < 0.08:
            return "A"
        # B
        if fingers_up == [0, 1, 1, 1, 1] and landmarks[THUMB_TIP][0] < landmarks[INDEX_MCP][0]:
            return "B"
        # C
        if all(is_finger_bent(t, d, p) for t, d, p in [
            (INDEX_TIP, INDEX_DIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP),
            (RING_TIP, RING_DIP, RING_PIP),
            (PINKY_TIP, PINKY_DIP, PINKY_PIP)
        ]) and thumb_index_distance > 0.05:
            return "C"
        # D
        if fingers_up == [0, 1, 0, 0, 0] and thumb_middle_distance < 0.05:
            return "D"
        # E
        if all(f == 0 for f in fingers_up) and landmarks[THUMB_TIP][1] > landmarks[INDEX_PIP][1]:
            return "E"
        # F
        if (fingers_up[3] == 1 and fingers_up[4] == 1 and
                thumb_index_distance < 0.05 and fingers_up[1] == 0):
            return "F"
        # G
        if (fingers_up[1] == 1 and fingers_up[0] == 1 and
                all(f == 0 for f in fingers_up[2:]) and
                abs(landmarks[INDEX_TIP][1] - landmarks[THUMB_TIP][1]) < 0.05):
            return "G"
        # H
        if (fingers_up[1] == 1 and fingers_up[2] == 1 and
                all(f == 0 for f in fingers_up[3:]) and fingers_up[0] == 0 and
                abs(landmarks[INDEX_TIP][1] - landmarks[MIDDLE_TIP][1]) < 0.03):
            return "H"
        # I / J
        if fingers_up == [0, 0, 0, 0, 1] and landmarks[THUMB_TIP][0] < landmarks[MIDDLE_MCP][0]:
            return "I"
        if fingers_up == [0, 0, 0, 0, 1]:
            return "J (static)"
        # K
        if (fingers_up[1] == 1 and fingers_up[2] == 1 and
                all(f == 0 for f in fingers_up[3:]) and
                self._get_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_PIP]) < 0.05):
            return "K"
        # L
        if fingers_up == [1, 1, 0, 0, 0]:
            angle = self.calculate_angle(landmarks[THUMB_TIP][:2], landmarks[INDEX_MCP][:2], landmarks[INDEX_TIP][:2])
            if 70 < angle < 110:
                return "L"
        # M / N
        if all(f == 0 for f in fingers_up[1:4]) and landmarks[THUMB_TIP][1] > landmarks[RING_DIP][1]:
            return "M"
        if all(f == 0 for f in fingers_up[1:3]) and fingers_up[3] == 0 and landmarks[THUMB_TIP][1] > landmarks[MIDDLE_DIP][1]:
            return "N"
        # O
        if thumb_index_distance < 0.05 and all(is_finger_bent(t, d, p) for t, d, p in [
            (INDEX_TIP, INDEX_DIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP),
            (RING_TIP, RING_DIP, RING_PIP),
            (PINKY_TIP, PINKY_DIP, PINKY_PIP)
        ]):
            return "O"
        # P / Q / R / S / T / U / V / W / X / Y / Z (brief heuristics)
        if fingers_up[1] == 1 and fingers_up[2] == 1 and landmarks[INDEX_TIP][1] > landmarks[WRIST][1] and self._get_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_PIP]) < 0.05:
            return "P"
        if fingers_up[1] == 1 and fingers_up[0] == 1 and landmarks[INDEX_TIP][1] > landmarks[WRIST][1]:
            return "Q"
        if fingers_up[1] == 1 and fingers_up[2] == 1 and abs(landmarks[INDEX_TIP][0] - landmarks[MIDDLE_TIP][0]) < 0.02:
            return "R"
        if all(f == 0 for f in fingers_up) and landmarks[THUMB_TIP][1] < landmarks[INDEX_PIP][1]:
            return "S"
        if all(f == 0 for f in fingers_up[1:]) and landmarks[THUMB_TIP][0] > landmarks[INDEX_MCP][0] and landmarks[THUMB_TIP][0] < landmarks[MIDDLE_MCP][0]:
            return "T"
        index_middle_distance = self._get_distance(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
        if fingers_up[1] == 1 and fingers_up[2] == 1 and index_middle_distance < 0.03 and all(f == 0 for f in fingers_up[3:]):
            return "U"

        if fingers_up[1] == 1 and fingers_up[2] == 1 and index_middle_distance > 0.05 and all(f == 0 for f in fingers_up[3:]):
            return "V"
        if fingers_up == [0, 1, 1, 1, 0]:
            return "W"
        if is_finger_bent(INDEX_TIP, INDEX_DIP, INDEX_PIP) and all(f == 0 for f in fingers_up[2:]):
            return "X"
        if fingers_up == [1, 0, 0, 0, 1] and thumb_pinky_distance > 0.1:
            return "Y"
        if fingers_up == [0, 1, 0, 0, 0]:
            return "Z (static)"

        return None

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        detected_letters: List[str] = []
        hand_centers: List[Tuple[int, int]] = []

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                letter = self.detect_asl_letter(hand_landmarks)
                if letter:
                    detected_letters.append(letter)

                # compute hand center (use middle_mcp 9)
                cx = int(hand_landmarks.landmark[9].x * w)
                cy = int(hand_landmarks.landmark[9].y * h)
                hand_centers.append((cx, cy))

                # store for motion-based detection and analytics
                self.landmark_history.append(hand_landmarks)

        return frame, detected_letters, hand_centers


# -------------------- Camera class --------------------
class Camera:
    def __init__(self, src: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.src)
        # try to set resolution; not guaranteed on all cameras
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        return self.cap.read()

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# -------------------- WordBuilder --------------------
class WordBuilder:
    def __init__(self, hold_frames: int = 8):
        self.current_word = ""
        self.last_letter: Optional[str] = None
        self.letter_count = 0
        self.hold_frames = hold_frames

    def update_with_detected(self, detected_letters: List[str]) -> None:
        if detected_letters:
            current_letter = detected_letters[0]
            if current_letter == self.last_letter:
                self.letter_count += 1
            else:
                self.letter_count = 1
                self.last_letter = current_letter
        else:
            # decay so it doesn't stick forever; keep last_letter so manual commit still possible
            self.letter_count = max(0, self.letter_count - 1)
            if self.letter_count == 0:
                self.last_letter = None

    def try_commit_letter(self) -> Optional[str]:
        if self.last_letter and self.letter_count >= self.hold_frames:
            letter = self.last_letter
            self.current_word += letter
            self.letter_count = 0
            self.last_letter = None
            return letter
        return None

    def clear(self) -> None:
        self.current_word = ""
        self.last_letter = None
        self.letter_count = 0


# -------------------- High-level App --------------------
class ASLApp:
    def __init__(self, detector: ASLDetector, camera: Camera, word_builder: WordBuilder):
        self.detector = detector
        self.camera = camera
        self.word_builder = word_builder
        self.fps_smooth = 0.0
        self.last_time = time.time()

    def run(self) -> None:
        self.camera.open()
        print("ASL Detection Started. Press 'q' to quit.")
        print("Press 'space' to add current letter to word, 'c' to clear word")

        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    break

                # mirror for natural webcam view
                frame = cv2.flip(frame, 1)

                processed_frame, detected_letters, hand_centers = self.detector.process_frame(frame)

                # update word builder
                self.word_builder.update_with_detected(detected_letters)

                # FPS calculation (smoothed)
                now = time.time()
                dt = now - self.last_time if now != self.last_time else 1/30
                fps = 1.0 / dt if dt > 0 else 0.0
                self.fps_smooth = 0.85 * self.fps_smooth + 0.15 * fps
                self.last_time = now

                h, w, _ = processed_frame.shape

                # Top translucent bar
                draw_transparent_rect(processed_frame, (0, 0), (w, 80), (30, 30, 30), alpha=0.55)
                put_text_with_shadow(processed_frame, "ASL OOP Detector", (18, 36), scale=1.0, thickness=2)
                put_text_with_shadow(processed_frame, f"FPS: {self.fps_smooth:.1f}", (w - 140, 36), scale=0.8, thickness=2)

                # Right-side panel
                panel_w = 300
                draw_transparent_rect(processed_frame, (w - panel_w, 80), (w, h), (20, 20, 30), alpha=0.5)
                put_text_with_shadow(processed_frame, "Status", (w - panel_w + 16, 110), scale=0.8)
                put_text_with_shadow(processed_frame, f"Detected hands: {len(hand_centers)}", (w - panel_w + 16, 140), scale=0.6)
                put_text_with_shadow(processed_frame, f"Word length: {len(self.word_builder.current_word)}", (w - panel_w + 16, 160), scale=0.6)
                put_text_with_shadow(processed_frame, "Controls:", (w - panel_w + 16, 195), scale=0.7)
                put_text_with_shadow(processed_frame, "Space: Add | C: Clear | Q: Quit", (w - panel_w + 16, 215), scale=0.45)

                # Center big word display (semi translucent)
                draw_transparent_rect(processed_frame, (10, h - 120), (w - panel_w - 10, h - 10), (10, 10, 10), alpha=0.45)
                put_text_with_shadow(processed_frame, f"Word: {self.word_builder.current_word}", (20, h - 70), scale=1.0, thickness=2)

                # Show the candidate letter big if any
                candidate = self.word_builder.last_letter if self.word_builder.last_letter else (detected_letters[0] if detected_letters else None)
                if candidate:
                    # big letter box
                    box_size = 220
                    box_x = (w - panel_w) // 2 - box_size // 2
                    box_y = 120
                    draw_transparent_rect(processed_frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (15, 70, 90), alpha=0.45)
                    # letter with shadow
                    put_text_with_shadow(processed_frame, candidate, (box_x + 40, box_y + 140), scale=4.0, thickness=4)
                    # hold progress bar
                    progress = min(1.0, (self.word_builder.letter_count / max(1, self.word_builder.hold_frames)))
                    draw_hold_bar(processed_frame, (box_x + 24, box_y + box_size - 26), (box_size - 48, 16), progress, bg_color=(50, 50, 60), fg_color=(0, 200, 200))

                # Pulsing circles around each detected hand center
                for i, (cx, cy) in enumerate(hand_centers):
                    # pulse depends on time and index for variety
                    pulse = 0.5 + 0.5 * math.sin(time.time() * 3 + i)
                    radius = int(20 + 20 * pulse)
                    thickness = 2
                    alpha = 0.35
                    overlay = processed_frame.copy()
                    cv2.circle(overlay, (cx, cy), radius, (0, 200, 200), thickness)
                    cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)

                # tiny detection labels under the biggest hand center if present
                if hand_centers and detected_letters:
                    cx, cy = hand_centers[0]
                    small_label = f"Letter: {detected_letters[0]}"
                    put_text_with_shadow(processed_frame, small_label, (cx - 60, cy + 60), scale=0.6, thickness=2)

                # Show frame
                cv2.imshow('ASL Detection (Attractive OOP UI)', processed_frame)

                # Keyboard handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    committed = self.word_builder.try_commit_letter()
                    if committed:
                        print(f"Added '{committed}' to word: {self.word_builder.current_word}")
                elif key == ord('c'):
                    self.word_builder.clear()
                    print("Word cleared")

        finally:
            self.camera.release()
            cv2.destroyAllWindows()


# -------------------- main --------------------
def main():
    detector = ASLDetector()
    camera = Camera(src=0, width=1280, height=720, fps=30)
    word_builder = WordBuilder(hold_frames=8)
    app = ASLApp(detector, camera, word_builder)
    app.run()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Error:", e)
        print("Make sure you have installed the required packages:")
        print("pip install opencv-python mediapipe numpy")
