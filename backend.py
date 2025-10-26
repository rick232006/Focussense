"""
FocusSense - Enhanced Python Backend
Advanced detection: Video content, phone distraction, improved eye tracking, stable PPG
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import json
import asyncio
import websockets
from datetime import datetime
import mss
from PIL import Image
import pytesseract  # OCR for text detection
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')


class VideoContentDetector:
    """Detects video/multimedia content on screen"""
    
    def __init__(self):
        self.frame_buffer = deque(maxlen=10)  # Store last 10 frames
        self.motion_threshold = 25.0
        self.video_confidence = 0.0
        
    def detect_video_content(self, screen_frame):
        """
        Detect if user is watching video content
        Uses motion detection between frames
        """
        try:
            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(screen_frame, cv2.COLOR_BGR2GRAY)
            
            self.frame_buffer.append(gray)
            
            if len(self.frame_buffer) < 2:
                return False, 0.0
            
            # Calculate motion between consecutive frames
            motion_scores = []
            for i in range(len(self.frame_buffer) - 1):
                diff = cv2.absdiff(self.frame_buffer[i], self.frame_buffer[i + 1])
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            avg_motion = np.mean(motion_scores)
            
            # High motion = likely video
            is_video = avg_motion > self.motion_threshold
            confidence = min(avg_motion / 50.0, 1.0)  # Normalize to 0-1
            
            self.video_confidence = confidence
            
            return is_video, confidence
            
        except Exception as e:
            print(f"[VideoDetector] Error: {e}")
            return False, 0.0


class PhoneDistractionDetector:
    """Detects when user is looking at phone using head pose and gaze direction"""
    
    def __init__(self):
        self.looking_down_history = deque(maxlen=30)  # 1 second at 30fps
        self.looking_away_history = deque(maxlen=30)
        self.phone_confidence = 0.0
        
    def detect_phone_usage(self, face_landmarks, frame_shape):
        """
        Detect phone distraction based on:
        1. Looking down (phone in lap)
        2. Looking away from screen
        3. Head tilt patterns
        """
        try:
            if face_landmarks is None:
                return False, 0.0
            
            h, w = frame_shape
            
            # Get key points for head pose
            nose_tip = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[152]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            forehead = face_landmarks.landmark[10]
            
            # Calculate head pitch (up/down angle)
            nose_y = nose_tip.y
            chin_y = chin.y
            forehead_y = forehead.y
            
            # Looking down if nose is close to chin (relative to forehead)
            vertical_ratio = (nose_y - forehead_y) / (chin_y - forehead_y)
            looking_down = vertical_ratio > 0.65  # Threshold for looking down
            
            # Calculate horizontal deviation (left/right)
            face_center_x = (left_eye.x + right_eye.x) / 2
            horizontal_deviation = abs(face_center_x - 0.5)
            looking_away = horizontal_deviation > 0.35  # Far from center
            
            self.looking_down_history.append(looking_down)
            self.looking_away_history.append(looking_away)
            
            # Phone usage if consistently looking down or away
            looking_down_ratio = sum(self.looking_down_history) / len(self.looking_down_history)
            looking_away_ratio = sum(self.looking_away_history) / len(self.looking_away_history)
            
            phone_probability = max(looking_down_ratio, looking_away_ratio * 0.7)
            is_using_phone = phone_probability > 0.6
            
            self.phone_confidence = phone_probability
            
            return is_using_phone, phone_probability
            
        except Exception as e:
            print(f"[PhoneDetector] Error: {e}")
            return False, 0.0


class EnhancedContentAnalyzer:
    """Advanced screen content analysis"""
    
    def __init__(self):
        self.video_detector = VideoContentDetector()
        self.last_analysis = {
            'category': 'Unknown',
            'productivity_score': 50,
            'is_video': False,
            'video_confidence': 0.0,
            'has_text': False,
            'brightness': 128
        }
        
        # Productivity scores
        self.productivity_map = {
            'Code/IDE': 95,
            'Document/Reading': 90,
            'Spreadsheet': 85,
            'Video/YouTube': 30,  # Low productivity
            'Social Media': 20,   # Very low
            'Mixed Content': 60,
            'Unknown': 50
        }
        
    def analyze_content(self, screen_frame, gaze_x, gaze_y):
        """
        Enhanced content analysis with video detection
        """
        try:
            h, w = screen_frame.shape[:2]
            
            # Detect video content (motion analysis)
            is_video, video_conf = self.video_detector.detect_video_content(screen_frame)
            
            # Extract gaze region
            x1 = max(0, gaze_x - 150)
            y1 = max(0, gaze_y - 150)
            x2 = min(w, gaze_x + 150)
            y2 = min(h, gaze_y + 150)
            
            gaze_region = screen_frame[y1:y2, x1:x2]
            
            # Analyze brightness
            avg_color = cv2.mean(gaze_region)[:3]
            brightness = sum(avg_color) / 3
            
            # Detect edges (more edges = more content/text)
            gray_region = cv2.cvtColor(gaze_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Determine category
            if is_video and video_conf > 0.4:
                category = 'Video/YouTube'
                productivity = self.productivity_map['Video/YouTube']
            elif brightness < 80 and edge_density > 0.15:
                # Dark with lots of edges = code
                category = 'Code/IDE'
                productivity = self.productivity_map['Code/IDE']
            elif brightness > 200 and edge_density > 0.1:
                # Bright with text
                category = 'Document/Reading'
                productivity = self.productivity_map['Document/Reading']
            elif brightness > 180 and edge_density < 0.08:
                # Bright, low edges = possibly social media
                category = 'Social Media'
                productivity = self.productivity_map['Social Media']
            elif 120 < brightness < 180:
                category = 'Mixed Content'
                productivity = self.productivity_map['Mixed Content']
            else:
                category = 'Unknown'
                productivity = self.productivity_map['Unknown']
            
            self.last_analysis = {
                'category': category,
                'productivity_score': productivity,
                'is_video': is_video,
                'video_confidence': video_conf,
                'brightness': brightness,
                'edge_density': edge_density
            }
            
            return self.last_analysis
            
        except Exception as e:
            print(f"[ContentAnalyzer] Error: {e}")
            return self.last_analysis


class UltraSensitiveEyeTracker:
    """Ultra-sensitive eye tracking with improved accuracy"""
    
    def __init__(self, screen_width=1920, screen_height=1080):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.8,  # Even higher
            min_tracking_confidence=0.8
        )
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Advanced smoothing
        self.gaze_history = deque(maxlen=7)  # 7-frame smoothing
        self.velocity_buffer = deque(maxlen=5)
        
        # Calibration
        self.calibration_offset_x = 0.0
        self.calibration_offset_y = 0.0
        
        # Eye landmarks
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
    def process_frame(self, frame):
        """Ultra-sensitive eye tracking"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Extract iris positions with high precision
            left_iris_pts = np.array([(landmarks.landmark[i].x * w,
                                       landmarks.landmark[i].y * h)
                                      for i in self.LEFT_IRIS], dtype=np.float64)
            right_iris_pts = np.array([(landmarks.landmark[i].x * w,
                                        landmarks.landmark[i].y * h)
                                       for i in self.RIGHT_IRIS], dtype=np.float64)
            
            # Sub-pixel iris centers
            left_iris = np.mean(left_iris_pts, axis=0)
            right_iris = np.mean(right_iris_pts, axis=0)
            
            # Weighted average (slightly favor dominant eye)
            gaze_x_cam = left_iris[0] * 0.5 + right_iris[0] * 0.5
            gaze_y_cam = left_iris[1] * 0.5 + right_iris[1] * 0.5
            
            # Apply calibration offset
            gaze_x_cam += self.calibration_offset_x
            gaze_y_cam += self.calibration_offset_y
            
            # Advanced smoothing with velocity prediction
            current_gaze = np.array([gaze_x_cam, gaze_y_cam])
            
            if len(self.gaze_history) > 0:
                velocity = current_gaze - self.gaze_history[-1]
                self.velocity_buffer.append(velocity)
                
                # Predict next position
                if len(self.velocity_buffer) >= 3:
                    avg_velocity = np.mean(list(self.velocity_buffer), axis=0)
                    predicted = current_gaze + avg_velocity * 0.3
                    # Blend prediction with actual
                    current_gaze = current_gaze * 0.7 + predicted * 0.3
            
            self.gaze_history.append(current_gaze)
            
            # Smooth over history
            if len(self.gaze_history) >= 5:
                smoothed_gaze = np.mean(list(self.gaze_history)[-5:], axis=0)
                gaze_x_cam, gaze_y_cam = smoothed_gaze
            
            # Map to screen
            gaze_x_norm = np.clip(gaze_x_cam / w, 0, 1)
            gaze_y_norm = np.clip(gaze_y_cam / h, 0, 1)
            
            screen_x = int(gaze_x_norm * self.screen_width)
            screen_y = int(gaze_y_norm * self.screen_height)
            
            # Calculate EAR
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE, w, h)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            
            # Head pose
            nose = landmarks.landmark[1]
            chin = landmarks.landmark[152]
            forehead = landmarks.landmark[10]
            
            vertical_ratio = (nose.y - forehead.y) / (chin.y - forehead.y)
            looking_at_screen = 0.3 < vertical_ratio < 0.7
            
            return {
                'gaze_x': gaze_x_cam,
                'gaze_y': gaze_y_cam,
                'screen_x': screen_x,
                'screen_y': screen_y,
                'screen_x_norm': gaze_x_norm,
                'screen_y_norm': gaze_y_norm,
                'ear': ear,
                'left_iris': left_iris,
                'right_iris': right_iris,
                'looking_at_screen': looking_at_screen,
                'landmarks': landmarks,
                'confidence': 0.95
            }
            
        except Exception as e:
            print(f"[EyeTracker] Error: {e}")
            return None
    
    def _calculate_ear(self, landmarks, eye_indices, w, h):
        """Calculate Eye Aspect Ratio"""
        pts = [(int(landmarks.landmark[i].x * w),
                int(landmarks.landmark[i].y * h))
               for i in eye_indices]
        
        if len(pts) < 6:
            return 0.3
        
        v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        h_dist = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        
        if h_dist == 0:
            return 0.3
        
        return (v1 + v2) / (2.0 * h_dist)


class StablePPGProcessor:
    """Stable heart rate detection with filtering and outlier removal"""
    
    def __init__(self, fps=30, window_size=12):
        self.fps = fps
        self.window_size = window_size
        self.buffer_size = fps * window_size
        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.hr_history = deque(maxlen=10)  # Last 10 measurements
        self.last_stable_hr = 70
        
    def extract_signal(self, frame, landmarks):
        """Extract PPG signal from forehead"""
        if landmarks is None:
            return None
        
        try:
            h, w = frame.shape[:2]
            
            # Forehead region
            forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]
            
            forehead_pts = np.array([[int(landmarks.landmark[i].x * w),
                                      int(landmarks.landmark[i].y * h)]
                                     for i in forehead_indices], dtype=np.int32)
            
            # Create mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [forehead_pts], 255)
            
            # Extract green channel (best for PPG)
            mean_val = cv2.mean(frame, mask=mask)[1]
            
            return mean_val
            
        except Exception as e:
            return None
    
    def process_signal(self, signal_value):
        """Process PPG signal with stability filtering"""
        if signal_value is None:
            return self.last_stable_hr
        
        self.signal_buffer.append(signal_value)
        
        if len(self.signal_buffer) < self.buffer_size:
            return self.last_stable_hr
        
        try:
            # Convert to array
            signal_array = np.array(self.signal_buffer)
            
            # Detrend (remove DC component)
            signal_array = signal_array - np.mean(signal_array)
            
            # Bandpass filter (0.7-3 Hz = 42-180 BPM)
            nyquist = self.fps / 2.0
            low = 0.7 / nyquist
            high = 3.0 / nyquist
            
            # Design butterworth filter
            b, a = signal.butter(3, [low, high], btype='band')
            filtered_signal = signal.filtfilt(b, a, signal_array)
            
            # FFT
            fft_vals = np.abs(np.fft.fft(filtered_signal))
            fft_freq = np.fft.fftfreq(len(filtered_signal), 1.0 / self.fps)
            
            # Get positive frequencies
            pos_mask = fft_freq > 0
            fft_freq = fft_freq[pos_mask]
            fft_vals = fft_vals[pos_mask]
            
            # Find peak in valid range
            valid_mask = (fft_freq >= 0.7) & (fft_freq <= 3.0)
            
            if np.sum(valid_mask) == 0:
                return self.last_stable_hr
            
            valid_fft = fft_vals[valid_mask]
            valid_freq = fft_freq[valid_mask]
            
            # Find highest peak
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freq[peak_idx]
            heart_rate = peak_freq * 60
            
            # Validate heart rate
            if 45 <= heart_rate <= 180:
                self.hr_history.append(heart_rate)
                
                # Remove outliers using IQR method
                if len(self.hr_history) >= 5:
                    hr_array = np.array(list(self.hr_history))
                    q1 = np.percentile(hr_array, 25)
                    q3 = np.percentile(hr_array, 75)
                    iqr = q3 - q1
                    
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filter outliers
                    filtered_hrs = hr_array[(hr_array >= lower_bound) & (hr_array <= upper_bound)]
                    
                    if len(filtered_hrs) > 0:
                        self.last_stable_hr = np.median(filtered_hrs)
            
            return int(self.last_stable_hr)
            
        except Exception as e:
            print(f"[PPG] Error: {e}")
            return self.last_stable_hr


class StrictConcentrationAnalyzer:
    """Strict concentration analysis - harsh on distractions"""
    
    def __init__(self):
        self.gaze_history = deque(maxlen=90)
        self.blink_history = deque(maxlen=900)
        self.screen_attention_history = deque(maxlen=90)
        self.phone_detection_history = deque(maxlen=60)
        
        self.EAR_THRESHOLD = 0.22
        self.blink_counter = 0
        
    def update(self, eye_data, content_analysis, phone_distraction, heart_rate):
        """
        Strict concentration with harsh penalties for:
        - Video content
        - Phone usage
        - Looking away
        - Poor content
        """
        if eye_data is None:
            return 25  # Very low if no face detected
        
        # Track data
        self.gaze_history.append((eye_data['screen_x'], eye_data['screen_y']))
        
        if eye_data['ear'] < self.EAR_THRESHOLD:
            self.blink_counter += 1
        self.blink_history.append(eye_data['ear'])
        
        looking_at_screen = eye_data.get('looking_at_screen', True)
        self.screen_attention_history.append(looking_at_screen)
        
        # Calculate base metrics
        gaze_stability = self._calculate_gaze_stability()
        screen_attention = self._calculate_screen_attention()
        blink_score = self._calculate_blink_score()
        
        # Content productivity (with harsh penalties)
        if content_analysis:
            productivity = content_analysis['productivity_score']
            
            # HARSH penalty for video content
            if content_analysis['is_video'] and content_analysis['video_confidence'] > 0.3:
                productivity *= 0.3  # 70% reduction!
            
        else:
            productivity = 50
        
        # Phone distraction penalty
        phone_penalty = 1.0
        if phone_distraction['is_distracted']:
            # Severe penalty for phone usage
            phone_penalty = 0.4  # 60% reduction!
            self.phone_detection_history.append(True)
        else:
            self.phone_detection_history.append(False)
        
        # Calculate base concentration
        base_concentration = (
            gaze_stability * 0.20 +
            screen_attention * 0.25 +
            productivity * 0.35 +
            blink_score * 0.10 +
            (heart_rate / 180 * 100) * 0.10
        )
        
        # Apply phone penalty
        final_concentration = base_concentration * phone_penalty
        
        # Additional penalty if consistently on phone
        if len(self.phone_detection_history) >= 30:
            phone_ratio = sum(self.phone_detection_history) / len(self.phone_detection_history)
            if phone_ratio > 0.5:
                final_concentration *= 0.5  # Another 50% cut!
        
        return int(np.clip(final_concentration, 0, 100))
    
    def _calculate_gaze_stability(self):
        """Gaze stability (stricter)"""
        if len(self.gaze_history) < 30:
            return 60
        
        recent = np.array(list(self.gaze_history)[-30:])
        std_x = np.std(recent[:, 0])
        std_y = np.std(recent[:, 1])
        
        variance = (std_x + std_y) / 2
        stability = max(0, 100 - variance * 1.2)  # Harsher penalty
        return stability
    
    def _calculate_screen_attention(self):
        """How much looking at screen"""
        if len(self.screen_attention_history) < 10:
            return 70
        
        attention_ratio = sum(self.screen_attention_history) / len(self.screen_attention_history)
        return attention_ratio * 100
    
    def _calculate_blink_score(self):
        """Blink rate scoring"""
        if len(self.blink_history) < 150:
            return 70
        
        seconds = len(self.blink_history) / 30.0
        bpm = (self.blink_counter / seconds) * 60
        
        if 12 <= bpm <= 20:
            return 100
        elif bpm < 12:
            return 60 + (bpm / 12) * 40
        else:
            return max(30, 100 - (bpm - 20) * 5)


class BackendProcessor:
    """Main backend processor"""
    
    def __init__(self):
        # Get screen size
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            self.screen_width = monitor['width']
            self.screen_height = monitor['height']
        
        # Initialize components
        self.eye_tracker = UltraSensitiveEyeTracker(self.screen_width, self.screen_height)
        self.content_analyzer = EnhancedContentAnalyzer()
        self.phone_detector = PhoneDistractionDetector()
        self.ppg_processor = StablePPGProcessor()
        self.concentration_analyzer = StrictConcentrationAnalyzer()
        
        # State
        self.running = False
        self.camera = None
        self.current_screen_frame = None
        
    def start(self):
        """Start processing"""
        self.running = True
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        
        print("[Backend] Started with enhanced detection")
        print(f"[Backend] Screen: {self.screen_width}x{self.screen_height}")
    
    def capture_screen(self):
        """Capture screen"""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                screen_array = np.array(img)
                screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
                screen_array = cv2.resize(screen_array, (1280, 720))
                self.current_screen_frame = screen_array
        except Exception as e:
            print(f"[Backend] Screen capture error: {e}")
    
    def process_frame(self):
        """Process single frame"""
        if not self.camera or not self.running:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        # Process eye tracking
        eye_data = self.eye_tracker.process_frame(frame)
        
        # Capture screen periodically
        if int(time.time() * 2) % 2 == 0:  # Every 0.5 seconds
            self.capture_screen()
        
        # Analyze content
        content_analysis = None
        if eye_data and self.current_screen_frame is not None:
            content_analysis = self.content_analyzer.analyze_content(
                self.current_screen_frame,
                eye_data['screen_x'],
                eye_data['screen_y']
            )
        
        # Detect phone distraction
        phone_distraction = {'is_distracted': False, 'confidence': 0.0}
        if eye_data:
            is_phone, phone_conf = self.phone_detector.detect_phone_usage(
                eye_data['landmarks'],
                frame.shape[:2]
            )
            phone_distraction = {'is_distracted': is_phone, 'confidence': phone_conf}
        
        # Process PPG
        heart_rate = 0
        if eye_data:
            signal = self.ppg_processor.extract_signal(frame, eye_data['landmarks'])
            heart_rate = self.ppg_processor.process_signal(signal)
        
        # Calculate concentration (STRICT)
        concentration = self.concentration_analyzer.update(
            eye_data,
            content_analysis,
            phone_distraction,
            heart_rate
        )
        
        # Prepare visualization frame
        if eye_data:
            # Draw gaze
            gaze_x = int(eye_data['gaze_x'])
            gaze_y = int(eye_data['gaze_y'])
            cv2.drawMarker(frame, (gaze_x, gaze_y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 20, 2)
            cv2.circle(frame, (gaze_x, gaze_y), 10, (0, 255, 0), 2)
            
            # Draw iris
            cv2.circle(frame, tuple(map(int, eye_data['left_iris'])), 5, (255, 120, 0), -1)
            cv2.circle(frame, tuple(map(int, eye_data['right_iris'])), 5, (255, 120, 0), -1)
        
        # Convert frame to base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = buffer.tobytes()
        
        # Package data
        result = {
            'timestamp': time.time(),
            'concentration': concentration,
            'heart_rate': heart_rate,
            'frame': frame_base64.hex(),  # Convert to hex for JSON
            'eye_data': {
                'detected': eye_data is not None,
                'screen_x': eye_data['screen_x'] if eye_data else 0,
                'screen_y': eye_data['screen_y'] if eye_data else 0,
                'looking_at_screen': eye_data['looking_at_screen'] if eye_data else False
            } if eye_data else {'detected': False},
            'content': content_analysis if content_analysis else {
                'category': 'Unknown',
                'productivity_score': 50,
                'is_video': False
            },
            'phone_distraction': phone_distraction,
            'blink_rate': int(self.concentration_analyzer.blink_counter / 
                            max(1, len(self.concentration_analyzer.blink_history) / 30.0))
        }
        
        return result
    
    def stop(self):
        """Stop processing"""
        self.running = False
        if self.camera:
            self.camera.release()
        print("[Backend] Stopped")


# WebSocket server
async def handle_client(websocket, path):
    """Handle WebSocket client connection"""
    print(f"[WebSocket] Client connected: {path}")
    
    processor = BackendProcessor()
    processor.start()
    
    try:
        # Send data loop
        while processor.running:
            data = processor.process_frame()
            
            if data:
                # Send as JSON
                await websocket.send(json.dumps(data))
            
            await asyncio.sleep(0.033)  # ~30 FPS
            
    except websockets.exceptions.ConnectionClosed:
        print("[WebSocket] Client disconnected")
    finally:
        processor.stop()


async def start_server():
    """Start WebSocket server"""
    print("[Server] Starting on ws://localhost:8765")
    
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        print("="*70)
        print("FocusSense Enhanced Python Backend")
        print("="*70)
        print("Features:")
        print("  • Ultra-sensitive eye tracking (0.8 confidence)")
        print("  • Video content detection (motion analysis)")
        print("  • Phone distraction detection (head pose)")
        print("  • Stable PPG heart rate (IQR filtering)")
        print("  • Strict concentration scoring")
        print("="*70)
        
        asyncio.run(start_server())
        
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
    except Exception as e:
        print(f"\n[Server] Error: {e}")
        import traceback
        traceback.print_exc()
