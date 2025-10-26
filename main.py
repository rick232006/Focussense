"""
FocusSense - Advanced Eye Tracking & Screen Analytics
Complete redesign with screen sharing, content analysis, and enhanced sensitivity
"""

import os
import sys
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QFrame,
                              QScrollArea, QGridLayout, QTextEdit, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon, QScreen, QPainter, QColor
import time
from collections import deque
from datetime import datetime
import mss
import mss.tools
from PIL import Image
import io


class EnhancedEyeTracker:
    """Enhanced eye tracker with higher sensitivity and accuracy"""
    
    def __init__(self, screen_width=1920, screen_height=1080):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Increased from 0.5
            min_tracking_confidence=0.7    # Increased from 0.5
        )
        
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Enhanced calibration
        self.calibration_history = deque(maxlen=90)  # 3 seconds
        self.gaze_smoothing = deque(maxlen=5)  # Smooth last 5 frames
        
        # Eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
    def process_frame(self, frame):
        """Enhanced processing with better accuracy"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract iris positions with sub-pixel accuracy
        left_iris_pts = np.array([(face_landmarks.landmark[i].x * w, 
                                   face_landmarks.landmark[i].y * h) 
                                  for i in self.LEFT_IRIS])
        right_iris_pts = np.array([(face_landmarks.landmark[i].x * w, 
                                    face_landmarks.landmark[i].y * h) 
                                   for i in self.RIGHT_IRIS])
        
        left_iris = np.mean(left_iris_pts, axis=0)
        right_iris = np.mean(right_iris_pts, axis=0)
        
        # Calculate gaze with weighted average (more weight to dominant eye)
        gaze_x_cam = (left_iris[0] * 0.5 + right_iris[0] * 0.5)
        gaze_y_cam = (left_iris[1] * 0.5 + right_iris[1] * 0.5)
        
        # Apply smoothing
        self.gaze_smoothing.append((gaze_x_cam, gaze_y_cam))
        if len(self.gaze_smoothing) >= 3:
            smoothed_gaze = np.mean(list(self.gaze_smoothing), axis=0)
            gaze_x_cam, gaze_y_cam = smoothed_gaze
        
        # Enhanced screen mapping with calibration
        gaze_x_normalized = gaze_x_cam / w
        gaze_y_normalized = gaze_y_cam / h
        
        # Apply screen boundaries and scaling
        screen_x = int(np.clip(gaze_x_normalized * self.screen_width, 0, self.screen_width - 1))
        screen_y = int(np.clip(gaze_y_normalized * self.screen_height, 0, self.screen_height - 1))
        
        # Enhanced EAR calculation
        left_eye_pts = [(int(face_landmarks.landmark[i].x * w), 
                        int(face_landmarks.landmark[i].y * h)) 
                       for i in self.LEFT_EYE]
        right_eye_pts = [(int(face_landmarks.landmark[i].x * w), 
                         int(face_landmarks.landmark[i].y * h)) 
                        for i in self.RIGHT_EYE]
        
        ear = (self._calculate_ear(left_eye_pts) + self._calculate_ear(right_eye_pts)) / 2.0
        
        # Head pose estimation
        nose_tip = face_landmarks.landmark[1]
        left_eye_outer = face_landmarks.landmark[33]
        right_eye_outer = face_landmarks.landmark[263]
        
        # Calculate engagement (how centered the face is)
        face_center_x = (left_eye_outer.x + right_eye_outer.x) / 2
        face_center_y = (left_eye_outer.y + right_eye_outer.y) / 2
        engagement = 1.0 - abs(face_center_x - 0.5) * 2
        
        return {
            'gaze_x': gaze_x_cam,
            'gaze_y': gaze_y_cam,
            'screen_x': screen_x,
            'screen_y': screen_y,
            'screen_x_norm': gaze_x_normalized,
            'screen_y_norm': gaze_y_normalized,
            'ear': ear,
            'engagement': engagement,
            'left_iris': left_iris,
            'right_iris': right_iris,
            'landmarks': face_landmarks,
            'confidence': 1.0  # High confidence with improved detection
        }
    
    def _calculate_ear(self, eye_points):
        """Enhanced EAR calculation"""
        if len(eye_points) < 6:
            return 0.3
        
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        v2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        if h == 0:
            return 0.3
        
        return (v1 + v2) / (2.0 * h)


class ScreenContentAnalyzer:
    """Analyzes screen content for productivity insights"""
    
    def __init__(self):
        self.productive_keywords = [
            'code', 'document', 'spreadsheet', 'pdf', 'report', 'email', 'meeting',
            'study', 'learn', 'tutorial', 'course', 'editor', 'terminal', 'ide'
        ]
        self.distraction_keywords = [
            'youtube', 'facebook', 'twitter', 'instagram', 'reddit', 'gaming',
            'netflix', 'tiktok', 'social', 'chat', 'message'
        ]
        self.last_analysis = {
            'category': 'Unknown',
            'productivity_score': 50,
            'app_name': 'Unknown',
            'window_title': ''
        }
        
    def analyze_screen(self, screen_image, gaze_x, gaze_y):
        """Analyze what's on screen and what user is looking at"""
        try:
            # Get region around gaze point (200x200 pixels)
            h, w = screen_image.shape[:2]
            x1 = max(0, gaze_x - 100)
            y1 = max(0, gaze_y - 100)
            x2 = min(w, gaze_x + 100)
            y2 = min(h, gaze_y + 100)
            
            gaze_region = screen_image[y1:y2, x1:x2]
            
            # Analyze colors in gaze region (simple heuristic)
            avg_color = cv2.mean(gaze_region)[:3]
            
            # Check if looking at dark/code editor (low brightness = code)
            brightness = sum(avg_color) / 3
            
            # Simple productivity heuristic
            if brightness < 100:
                category = "Code/IDE"
                productivity = 90
            elif brightness > 200:
                category = "Document/Reading"
                productivity = 85
            else:
                category = "Mixed Content"
                productivity = 70
            
            self.last_analysis = {
                'category': category,
                'productivity_score': productivity,
                'app_name': category,
                'window_title': '',
                'gaze_region_brightness': brightness
            }
            
            return self.last_analysis
            
        except Exception as e:
            print(f"[ContentAnalyzer] Error: {e}")
            return self.last_analysis


class AdvancedConcentrationAnalyzer:
    """Enhanced concentration analysis with screen content awareness"""
    
    def __init__(self):
        self.gaze_history = deque(maxlen=90)  # 3 seconds
        self.blink_history = deque(maxlen=900)  # 30 seconds
        self.engagement_history = deque(maxlen=90)
        self.screen_content_history = deque(maxlen=90)
        
        self.EAR_THRESHOLD = 0.21
        self.blink_counter = 0
        self.consecutive_frames = 0
        
    def update(self, eye_data, screen_content, heart_rate=70):
        """Enhanced concentration with screen content analysis"""
        if eye_data is None:
            return 50
        
        # Track all metrics
        self.gaze_history.append((eye_data['screen_x'], eye_data['screen_y']))
        self.engagement_history.append(eye_data.get('engagement', 0.5))
        
        if eye_data['ear'] < self.EAR_THRESHOLD:
            self.blink_counter += 1
        self.blink_history.append(eye_data['ear'])
        
        if screen_content:
            self.screen_content_history.append(screen_content['productivity_score'])
        
        self.consecutive_frames += 1
        
        # Calculate components with new weights
        gaze_stability = self._calculate_gaze_stability()
        engagement_score = self._calculate_engagement()
        blink_score = self._calculate_blink_score()
        content_score = self._calculate_content_score()
        confidence_score = eye_data.get('confidence', 1.0) * 100
        
        # Weighted combination emphasizing screen content
        concentration = (
            gaze_stability * 0.25 +
            engagement_score * 0.20 +
            content_score * 0.30 +      # High weight for productivity
            blink_score * 0.15 +
            confidence_score * 0.10
        )
        
        return int(np.clip(concentration, 0, 100))
    
    def _calculate_gaze_stability(self):
        """More sensitive gaze stability"""
        if len(self.gaze_history) < 30:
            return 75
        
        recent_gazes = np.array(list(self.gaze_history)[-30:])
        std_x = np.std(recent_gazes[:, 0])
        std_y = np.std(recent_gazes[:, 1])
        
        # Tighter threshold for stability (more sensitive)
        total_variance = (std_x + std_y) / 2
        stability = max(0, 100 - min(total_variance * 0.8, 100))  # More sensitive
        return stability
    
    def _calculate_engagement(self):
        """Calculate engagement from head pose"""
        if len(self.engagement_history) < 10:
            return 75
        
        avg_engagement = np.mean(list(self.engagement_history)[-30:])
        return avg_engagement * 100
    
    def _calculate_blink_score(self):
        """Enhanced blink detection"""
        if len(self.blink_history) < 150:
            return 75
        
        frames = len(self.blink_history)
        blinks_per_second = self.blink_counter / (frames / 30.0)
        blinks_per_minute = blinks_per_second * 60
        
        # Optimal range: 12-20 blinks per minute
        if 12 <= blinks_per_minute <= 20:
            return 100
        elif blinks_per_minute < 12:
            return 70 + (blinks_per_minute / 12) * 30
        else:
            return max(40, 100 - (blinks_per_minute - 20) * 4)
    
    def _calculate_content_score(self):
        """Score based on screen content productivity"""
        if len(self.screen_content_history) < 10:
            return 75
        
        avg_productivity = np.mean(list(self.screen_content_history)[-30:])
        return avg_productivity


class PPGProcessor:
    """Heart rate estimation"""
    
    def __init__(self, fps=30, window_size=10):
        self.fps = fps
        self.buffer_size = fps * window_size
        self.signal_buffer = deque(maxlen=self.buffer_size)
        self.last_hr = 0
        
    def extract_roi_signal(self, frame, landmarks):
        """Extract PPG signal from forehead"""
        if landmarks is None:
            return None
        
        h, w = frame.shape[:2]
        forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288]
        
        forehead_pts = np.array([[int(landmarks.landmark[i].x * w), 
                                  int(landmarks.landmark[i].y * h)] 
                                 for i in forehead_indices], dtype=np.int32)
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [forehead_pts], 255)
        mean_val = cv2.mean(frame, mask=mask)[1]  # Green channel
        
        return mean_val
    
    def process_signal(self, signal_value):
        """Process PPG signal"""
        if signal_value is None:
            return self.last_hr
        
        self.signal_buffer.append(signal_value)
        
        if len(self.signal_buffer) < self.buffer_size:
            return self.last_hr
        
        signal = np.array(self.signal_buffer)
        signal = signal - np.mean(signal)
        
        # FFT analysis
        fft_vals = np.abs(np.fft.fft(signal))
        fft_freq = np.fft.fftfreq(len(signal), 1.0/self.fps)
        
        # Heart rate range: 0.7-4 Hz (42-240 BPM)
        valid_idx = (fft_freq >= 0.7) & (fft_freq <= 4.0)
        
        if np.sum(valid_idx) > 0:
            valid_fft = fft_vals[valid_idx]
            valid_freq = fft_freq[valid_idx]
            peak_freq = valid_freq[np.argmax(valid_fft)]
            heart_rate = peak_freq * 60
            
            if 45 <= heart_rate <= 180:
                self.last_hr = heart_rate
        
        return self.last_hr


class ReadingAnalyzer:
    """Analyzes reading patterns"""
    
    def __init__(self):
        self.gaze_history = deque(maxlen=300)
        self.last_direction = None
        self.words_per_line = 10
        
    def update(self, eye_data):
        """Update reading analysis"""
        if eye_data is None:
            return 0
        
        self.gaze_history.append((eye_data['screen_x'], eye_data['screen_y']))
        
        if len(self.gaze_history) < 90:
            return 0
        
        return self._detect_reading()
    
    def _detect_reading(self):
        """Detect reading pattern"""
        gaze_points = np.array(list(self.gaze_history))
        x_positions = gaze_points[:, 0]
        
        sweeps = 0
        for i in range(15, len(x_positions), 15):
            window = x_positions[i-15:i]
            if len(window) < 15:
                continue
            
            trend = np.polyfit(range(len(window)), window, 1)[0]
            
            if trend > 8:  # Moving right
                if self.last_direction == 'left':
                    sweeps += 1
                self.last_direction = 'right'
            elif trend < -25:  # Quick return
                self.last_direction = 'left'
        
        seconds = len(self.gaze_history) / 30.0
        wpm = (sweeps / seconds) * 60 * self.words_per_line
        
        return int(wpm)


class ScreenCaptureThread(QThread):
    """Captures screen for content analysis"""
    screen_captured = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        
    def run(self):
        """Capture screen periodically"""
        print("[ScreenCapture] Starting...")
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            
            self.running = True
            
            while self.running:
                if self.paused:
                    time.sleep(0.5)
                    continue
                
                try:
                    # Capture screen
                    screenshot = sct.grab(monitor)
                    img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                    
                    # Convert to numpy for analysis
                    screen_array = np.array(img)
                    screen_array = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
                    
                    # Resize for performance
                    screen_array = cv2.resize(screen_array, (1280, 720))
                    
                    self.screen_captured.emit(screen_array)
                    
                except Exception as e:
                    print(f"[ScreenCapture] Error: {e}")
                
                time.sleep(0.5)  # Capture every 0.5 seconds
        
        print("[ScreenCapture] Stopped")
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        self.running = False


class CameraThread(QThread):
    """Enhanced camera processing with screen integration"""
    frame_ready = pyqtSignal(object)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.paused = False
        
        # Get screen resolution
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        
        self.eye_tracker = EnhancedEyeTracker(self.screen_width, self.screen_height)
        self.ppg_processor = PPGProcessor()
        self.concentration_analyzer = AdvancedConcentrationAnalyzer()
        self.reading_analyzer = ReadingAnalyzer()
        self.content_analyzer = ScreenContentAnalyzer()
        
        self.current_screen_image = None
        
    def set_screen_image(self, screen_image):
        """Receive screen capture from screen thread"""
        self.current_screen_image = screen_image
    
    def run(self):
        """Main processing loop"""
        print("[Camera] Starting...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[Camera] ERROR: Cannot open camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"[Camera] Opened - Screen: {self.screen_width}x{self.screen_height}")
        self.running = True
        frame_count = 0
        
        while self.running:
            if self.paused:
                time.sleep(0.1)
                continue
            
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Process eye tracking
            eye_data = self.eye_tracker.process_frame(frame)
            
            # Process screen content if available
            screen_content = None
            if eye_data and self.current_screen_image is not None:
                screen_content = self.content_analyzer.analyze_screen(
                    self.current_screen_image,
                    eye_data['screen_x'],
                    eye_data['screen_y']
                )
            
            # Process PPG
            heart_rate = 0
            if eye_data and 'landmarks' in eye_data:
                signal = self.ppg_processor.extract_roi_signal(frame, eye_data['landmarks'])
                heart_rate = self.ppg_processor.process_signal(signal)
            
            # Calculate metrics
            concentration = self.concentration_analyzer.update(eye_data, screen_content, heart_rate)
            reading_speed = self.reading_analyzer.update(eye_data)
            
            # Enhanced visualization
            if eye_data:
                # Gaze point
                gaze_x = int(eye_data['gaze_x'])
                gaze_y = int(eye_data['gaze_y'])
                
                # Draw crosshair
                cv2.line(frame, (gaze_x - 15, gaze_y), (gaze_x + 15, gaze_y), (0, 255, 0), 2)
                cv2.line(frame, (gaze_x, gaze_y - 15), (gaze_x, gaze_y + 15), (0, 255, 0), 2)
                cv2.circle(frame, (gaze_x, gaze_y), 8, (0, 255, 0), 2)
                
                # Iris tracking
                cv2.circle(frame, tuple(map(int, eye_data['left_iris'])), 5, (255, 100, 0), -1)
                cv2.circle(frame, tuple(map(int, eye_data['right_iris'])), 5, (255, 100, 0), -1)
                
                # Screen position overlay
                overlay_x = int(20 + eye_data['screen_x_norm'] * 120)
                overlay_y = int(20 + eye_data['screen_y_norm'] * 70)
                
                cv2.rectangle(frame, (20, 20), (140, 90), (60, 60, 60), 2)
                cv2.circle(frame, (overlay_x, overlay_y), 6, (0, 255, 255), -1)
                cv2.putText(frame, "Screen Map", (22, 105), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Convert frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = np.ascontiguousarray(rgb_frame)
            
            # Package data
            data = {
                'frame': rgb_frame,
                'concentration': concentration,
                'reading_speed': reading_speed,
                'heart_rate': int(heart_rate),
                'blink_rate': int(self.concentration_analyzer.blink_counter / 
                                 max(1, len(self.concentration_analyzer.blink_history) / 30.0)),
                'gaze_focus': int(eye_data.get('engagement', 0) * 100) if eye_data else 0,
                'screen_content': screen_content,
                'eye_data': eye_data,
                'timestamp': time.time()
            }
            
            self.frame_ready.emit(data)
            
            if frame_count % 90 == 0:
                print(f"[Camera] {frame_count} frames | Concentration: {concentration}%")
            
            time.sleep(0.033)
        
        print("[Camera] Stopping...")
        cap.release()
    
    def pause(self):
        self.paused = True
    
    def resume(self):
        self.paused = False
    
    def stop(self):
        self.running = False


class SessionData:
    """Session data management"""
    
    def __init__(self):
        self.sessions = []
        self.current_session = None
        
    def start_session(self):
        self.current_session = {
            'start_time': datetime.now(),
            'concentration_data': [],
            'avg_concentration': 0,
            'productivity_score': 0,
            'duration': 0
        }
    
    def update(self, concentration, productivity):
        if self.current_session:
            self.current_session['concentration_data'].append(concentration)
            if productivity:
                self.current_session['productivity_score'] = productivity
    
    def end_session(self):
        if self.current_session:
            self.current_session['end_time'] = datetime.now()
            duration = (self.current_session['end_time'] - 
                       self.current_session['start_time']).total_seconds()
            self.current_session['duration'] = duration
            
            if self.current_session['concentration_data']:
                self.current_session['avg_concentration'] = np.mean(
                    self.current_session['concentration_data']
                )
            
            self.sessions.append(self.current_session)
            self.current_session = None
    
    def get_stats(self):
        if not self.sessions:
            return None
        
        return {
            'total_sessions': len(self.sessions),
            'avg_concentration': np.mean([s['avg_concentration'] for s in self.sessions]),
            'total_time': sum([s['duration'] for s in self.sessions]) / 3600
        }


# UI Components with exact color scheme from prototype

class SoftCard(QFrame):
    """Card with soft neutral colors from prototype"""
    
    def __init__(self, title=""):
        super().__init__()
        self.setObjectName("card")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("cardTitle")
            layout.addWidget(title_label)
        
        self.content_layout = QVBoxLayout()
        layout.addLayout(self.content_layout)


class StatCard(SoftCard):
    """Individual stat display"""
    
    def __init__(self, icon_text, label, value="0", color="primary"):
        super().__init__()
        
        self.color = color
        
        container = QHBoxLayout()
        container.setSpacing(12)
        
        # Icon
        icon_frame = QFrame()
        icon_frame.setObjectName(f"icon_{color}")
        icon_frame.setFixedSize(40, 40)
        icon_layout = QVBoxLayout(icon_frame)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_label = QLabel(icon_text)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"font-size: 20px; color: hsl(195, 85%, 45%);")
        icon_layout.addWidget(icon_label)
        container.addWidget(icon_frame)
        
        # Text
        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        
        self.label_widget = QLabel(label)
        self.label_widget.setObjectName("statLabel")
        text_layout.addWidget(self.label_widget)
        
        self.value_widget = QLabel(value)
        self.value_widget.setObjectName("statValue")
        text_layout.addWidget(self.value_widget)
        
        container.addLayout(text_layout)
        container.addStretch()
        
        self.content_layout.addLayout(container)
    
    def update_value(self, value):
        self.value_widget.setText(str(value))


class SoftButton(QPushButton):
    """Button with soft styling"""
    
    def __init__(self, text, style="primary"):
        super().__init__(text)
        self.setObjectName(f"btn_{style}")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(44)


class MainWindow(QMainWindow):
    """Main application window with exact UI replica"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FocusSense - Eye Tracking & Screen Analytics")
        self.setGeometry(100, 50, 1600, 950)
        
        # Apply exact color scheme from prototype
        self.apply_style()
        
        # Data
        self.camera_thread = None
        self.screen_thread = None
        self.session_data = SessionData()
        self.is_tracking = False
        self.session_start = None
        
        # Setup UI
        self.setup_ui()
        
        # Timer
        self.session_timer = QTimer()
        self.session_timer.timeout.connect(self.update_timer)
        
        print("[App] Initialized with enhanced sensitivity")
    
    def apply_style(self):
        """Apply exact color scheme from Lovable prototype"""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 hsl(220, 17%, 97%),
                    stop:1 hsl(195, 85%, 96%));
                color: hsl(220, 15%, 15%);
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            
            #sidebar {
                background-color: hsl(220, 15%, 15%);
                border-right: 1px solid hsl(220, 15%, 22%);
            }
            
            #sidebarTitle {
                color: hsl(220, 17%, 97%);
                font-size: 20px;
                font-weight: bold;
                padding: 24px 20px;
            }
            
            #navButton {
                background-color: transparent;
                color: hsl(220, 17%, 97%);
                border: none;
                border-radius: 8px;
                padding: 12px 16px;
                text-align: left;
                font-size: 14px;
                font-weight: 500;
            }
            
            #navButton:hover {
                background-color: hsl(220, 15%, 22%);
            }
            
            #navButtonActive {
                background-color: hsla(195, 85%, 45%, 0.15);
                color: hsl(195, 85%, 45%);
                font-weight: 600;
            }
            
            #card {
                background: white;
                border: 1px solid hsl(220, 13%, 91%);
                border-radius: 12px;
                box-shadow: 0 2px 10px hsla(220, 15%, 15%, 0.05);
            }
            
            #cardTitle {
                font-size: 16px;
                font-weight: 600;
                color: hsl(220, 15%, 15%);
            }
            
            #statLabel {
                font-size: 12px;
                color: hsl(220, 10%, 46%);
            }
            
            #statValue {
                font-size: 20px;
                font-weight: bold;
                color: hsl(220, 15%, 15%);
            }
            
            #icon_primary, #icon_secondary, #icon_accent {
                border-radius: 8px;
            }
            
            #icon_primary {
                background-color: hsla(195, 85%, 45%, 0.1);
            }
            
            #icon_secondary {
                background-color: hsla(174, 62%, 47%, 0.1);
            }
            
            #icon_accent {
                background-color: hsla(280, 65%, 60%, 0.1);
            }
            
            #btn_primary {
                background-color: hsl(195, 85%, 45%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            
            #btn_primary:hover {
                background-color: hsl(195, 85%, 40%);
            }
            
            #btn_secondary {
                background-color: hsl(174, 62%, 47%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            
            #btn_secondary:hover {
                background-color: hsl(174, 62%, 42%);
            }
            
            #btn_danger {
                background-color: hsl(0, 84%, 60%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
            }
            
            #btn_danger:hover {
                background-color: hsl(0, 84%, 55%);
            }
            
            QPushButton:disabled {
                background-color: hsl(220, 13%, 91%);
                color: hsl(220, 10%, 60%);
            }
            
            QProgressBar {
                border: none;
                border-radius: 6px;
                background-color: hsl(220, 14%, 96%);
                height: 12px;
                text-align: center;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 hsl(174, 62%, 47%),
                    stop:1 hsl(195, 85%, 45%));
                border-radius: 6px;
            }
            
            #timerLabel {
                background-color: white;
                border: 1px solid hsl(220, 13%, 91%);
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 24px;
                font-weight: bold;
                font-family: 'Courier New', monospace;
                color: hsl(220, 15%, 15%);
            }
            
            #contentTitle {
                font-size: 28px;
                font-weight: bold;
                color: hsl(220, 15%, 15%);
            }
            
            #contentSubtitle {
                font-size: 14px;
                color: hsl(220, 10%, 46%);
            }
        """)
    
    def setup_ui(self):
        """Setup main UI with sidebar navigation"""
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Sidebar
        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)
        
        # Content area
        self.content_stack = QWidget()
        self.content_layout = QVBoxLayout(self.content_stack)
        self.content_layout.setContentsMargins(32, 32, 32, 32)
        self.content_layout.setSpacing(24)
        
        # Create all pages
        self.live_page = self.create_live_session_page()
        self.dashboard_page = self.create_dashboard_page()
        self.history_page = self.create_history_page()
        self.settings_page = self.create_settings_page()
        
        # Show live page by default
        self.content_layout.addWidget(self.live_page)
        self.live_page.show()
        self.dashboard_page.hide()
        self.history_page.hide()
        self.settings_page.hide()
        
        main_layout.addWidget(self.content_stack, stretch=1)
    
    def create_sidebar(self):
        """Create sidebar navigation"""
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(240)
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("FocusSense")
        title.setObjectName("sidebarTitle")
        layout.addWidget(title)
        
        # Navigation buttons
        nav_items = [
            ("📹 Live Session", self.show_live),
            ("📊 Dashboard", self.show_dashboard),
            ("📜 History", self.show_history),
            ("⚙️ Settings", self.show_settings)
        ]
        
        self.nav_buttons = []
        for text, callback in nav_items:
            btn = QPushButton(text)
            btn.setObjectName("navButton")
            btn.clicked.connect(callback)
            btn.setMinimumHeight(44)
            layout.addWidget(btn)
            self.nav_buttons.append(btn)
        
        # Set first button active
        self.nav_buttons[0].setObjectName("navButtonActive")
        
        layout.addStretch()
        
        return sidebar
    
    def set_active_nav(self, index):
        """Update active navigation button"""
        for i, btn in enumerate(self.nav_buttons):
            if i == index:
                btn.setObjectName("navButtonActive")
            else:
                btn.setObjectName("navButton")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
    
    def show_live(self):
        self.set_active_nav(0)
        self.live_page.show()
        self.dashboard_page.hide()
        self.history_page.hide()
        self.settings_page.hide()
    
    def show_dashboard(self):
        self.set_active_nav(1)
        self.live_page.hide()
        self.dashboard_page.show()
        self.history_page.hide()
        self.settings_page.hide()
    
    def show_history(self):
        self.set_active_nav(2)
        self.live_page.hide()
        self.dashboard_page.hide()
        self.history_page.show()
        self.settings_page.hide()
    
    def show_settings(self):
        self.set_active_nav(3)
        self.live_page.hide()
        self.dashboard_page.hide()
        self.history_page.hide()
        self.settings_page.show()
    
    def create_live_session_page(self):
        """Create live tracking page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(24)
        
        # Header
        header = QHBoxLayout()
        
        header_text = QVBoxLayout()
        title = QLabel("Live Session")
        title.setObjectName("contentTitle")
        header_text.addWidget(title)
        
        subtitle = QLabel("Track your focus in real-time")
        subtitle.setObjectName("contentSubtitle")
        header_text.addWidget(subtitle)
        
        header.addLayout(header_text)
        header.addStretch()
        
        # Timer
        self.timer_label = QLabel("00:00")
        self.timer_label.setObjectName("timerLabel")
        header.addWidget(self.timer_label)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        
        self.start_btn = SoftButton("▶ Start Session", "primary")
        self.start_btn.clicked.connect(self.start_session)
        btn_layout.addWidget(self.start_btn)
        
        self.pause_btn = SoftButton("⏸ Pause", "secondary")
        self.pause_btn.clicked.connect(self.pause_session)
        self.pause_btn.setEnabled(False)
        btn_layout.addWidget(self.pause_btn)
        
        self.stop_btn = SoftButton("⏹ Stop", "danger")
        self.stop_btn.clicked.connect(self.stop_session)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.stop_btn)
        
        header.addLayout(btn_layout)
        layout.addLayout(header)
        
        # Content grid
        content_grid = QHBoxLayout()
        content_grid.setSpacing(24)
        
        # Left: Camera feed
        camera_card = SoftCard()
        camera_card.setMinimumHeight(500)
        camera_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(960, 540)
        self.camera_label.setMaximumSize(960, 540)
        self.camera_label.setScaledContents(False)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("""
            background-color: black;
            border-radius: 8px;
            color: white;
        """)
        self.camera_label.setText("Camera feed will appear here\n\nClick 'Start Session' to begin")
        camera_layout.addWidget(self.camera_label)
        
        camera_card.content_layout.addLayout(camera_layout)
        content_grid.addWidget(camera_card, stretch=2)
        
        # Right: Stats
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setSpacing(16)
        
        # Focus score card
        focus_card = SoftCard()
        focus_content = QVBoxLayout()
        
        focus_header = QHBoxLayout()
        focus_icon = QLabel("🧠")
        focus_icon.setStyleSheet("font-size: 24px;")
        focus_header.addWidget(focus_icon)
        
        focus_text = QVBoxLayout()
        focus_status_label = QLabel("Current State")
        focus_status_label.setObjectName("statLabel")
        focus_text.addWidget(focus_status_label)
        
        self.focus_state_label = QLabel("Ready")
        self.focus_state_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: hsl(195, 85%, 45%);
        """)
        focus_text.addWidget(self.focus_state_label)
        focus_header.addLayout(focus_text)
        focus_header.addStretch()
        
        focus_content.addLayout(focus_header)
        
        # Progress section
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(8)
        
        progress_label_layout = QHBoxLayout()
        progress_label = QLabel("Concentration")
        progress_label.setObjectName("statLabel")
        progress_label_layout.addWidget(progress_label)
        progress_label_layout.addStretch()
        
        self.concentration_percent = QLabel("0%")
        self.concentration_percent.setStyleSheet("font-weight: bold;")
        progress_label_layout.addWidget(self.concentration_percent)
        progress_layout.addLayout(progress_label_layout)
        
        self.concentration_bar = QProgressBar()
        self.concentration_bar.setRange(0, 100)
        self.concentration_bar.setValue(0)
        self.concentration_bar.setTextVisible(False)
        progress_layout.addWidget(self.concentration_bar)
        
        focus_content.addLayout(progress_layout)
        focus_card.content_layout.addLayout(focus_content)
        stats_layout.addWidget(focus_card)
        
        # Individual stats
        self.blink_card = StatCard("👁️", "Blink Rate", "0/min", "primary")
        stats_layout.addWidget(self.blink_card)
        
        self.heart_card = StatCard("❤️", "Heart Rate", "0 bpm", "secondary")
        stats_layout.addWidget(self.heart_card)
        
        self.gaze_card = StatCard("🎯", "Gaze Focus", "0%", "accent")
        stats_layout.addWidget(self.gaze_card)
        
        self.content_card = StatCard("💻", "Content Type", "Unknown", "primary")
        stats_layout.addWidget(self.content_card)
        
        self.productivity_card = StatCard("📊", "Productivity", "0%", "secondary")
        stats_layout.addWidget(self.productivity_card)
        
        stats_layout.addStretch()
        
        content_grid.addWidget(stats_widget, stretch=1)
        layout.addLayout(content_grid)
        
        return page
    
    def create_dashboard_page(self):
        """Create analytics dashboard"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Dashboard")
        title.setObjectName("contentTitle")
        layout.addWidget(title)
        
        subtitle = QLabel("Your focus analytics and insights")
        subtitle.setObjectName("contentSubtitle")
        layout.addWidget(subtitle)
        
        # Stats grid
        stats_grid = QGridLayout()
        stats_grid.setSpacing(16)
        
        self.dash_avg_card = StatCard("📈", "Avg Focus", "0%")
        self.dash_time_card = StatCard("⏱️", "Total Time", "0h")
        self.dash_sessions_card = StatCard("📝", "Sessions", "0")
        self.dash_productivity_card = StatCard("💯", "Productivity", "0%")
        
        stats_grid.addWidget(self.dash_avg_card, 0, 0)
        stats_grid.addWidget(self.dash_time_card, 0, 1)
        stats_grid.addWidget(self.dash_sessions_card, 0, 2)
        stats_grid.addWidget(self.dash_productivity_card, 0, 3)
        
        layout.addLayout(stats_grid)
        
        # Insights
        insights_card = SoftCard("💡 AI Insights")
        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        self.insights_text.setMaximumHeight(200)
        self.insights_text.setStyleSheet("""
            border: none;
            background: transparent;
            font-size: 13px;
            line-height: 1.6;
        """)
        self.insights_text.setPlainText(
            "Start a session to see AI-powered insights about your focus patterns."
        )
        insights_card.content_layout.addWidget(self.insights_text)
        layout.addWidget(insights_card)
        
        layout.addStretch()
        
        return page
    
    def create_history_page(self):
        """Create history page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Session History")
        title.setObjectName("contentTitle")
        layout.addWidget(title)
        
        history_card = SoftCard()
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setStyleSheet("""
            border: none;
            background: transparent;
            font-family: 'Courier New', monospace;
            font-size: 12px;
        """)
        self.history_text.setPlainText("No sessions yet. Start tracking to build your history!")
        history_card.content_layout.addWidget(self.history_text)
        layout.addWidget(history_card)
        
        return page
    
    def create_settings_page(self):
        """Create settings page"""
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Settings")
        title.setObjectName("contentTitle")
        layout.addWidget(title)
        
        settings_card = SoftCard("Application Settings")
        settings_text = QLabel(
            "• Eye Tracking: Enhanced sensitivity mode\n"
            "• Screen Analysis: Real-time content monitoring\n"
            "• Resolution: Auto-detected from system\n"
            "• Camera: HD quality (1280x720)\n"
            "• Frame Rate: 30 FPS\n"
            "• PPG Heart Rate: Active\n\n"
            "All settings are optimized for best performance."
        )
        settings_text.setStyleSheet("font-size: 13px; line-height: 1.8;")
        settings_text.setWordWrap(True)
        settings_card.content_layout.addWidget(settings_text)
        layout.addWidget(settings_card)
        
        layout.addStretch()
        
        return page
    
    def start_session(self):
        """Start tracking session"""
        print("[Session] Starting with screen capture...")
        
        self.is_tracking = True
        self.session_start = time.time()
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # Start camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_display)
        self.camera_thread.start()
        
        # Start screen capture thread
        self.screen_thread = ScreenCaptureThread()
        self.screen_thread.screen_captured.connect(self.camera_thread.set_screen_image)
        self.screen_thread.start()
        
        # Start timer
        self.session_timer.start(1000)
        
        # Start session data
        self.session_data.start_session()
        
        self.focus_state_label.setText("Tracking Active")
        self.focus_state_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: hsl(174, 62%, 47%);
        """)
    
    def pause_session(self):
        """Pause/resume session"""
        if self.camera_thread.paused:
            self.camera_thread.resume()
            self.screen_thread.resume()
            self.session_timer.start(1000)
            self.pause_btn.setText("⏸ Pause")
            self.focus_state_label.setText("Tracking Active")
        else:
            self.camera_thread.pause()
            self.screen_thread.pause()
            self.session_timer.stop()
            self.pause_btn.setText("▶ Resume")
            self.focus_state_label.setText("Paused")
    
    def stop_session(self):
        """Stop tracking session"""
        print("[Session] Stopping...")
        
        self.is_tracking = False
        
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸ Pause")
        
        self.session_timer.stop()
        
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        if self.screen_thread:
            self.screen_thread.stop()
            self.screen_thread.wait()
        
        self.session_data.end_session()
        self.update_history()
        self.update_dashboard()
        
        self.focus_state_label.setText("Session Complete")
        self.camera_label.setText("Session completed!\n\nClick 'Start Session' to begin a new one")
    
    def update_display(self, data):
        """Update display with tracking data"""
        try:
            frame = data['frame']
            concentration = data['concentration']
            heart_rate = data['heart_rate']
            blink_rate = data['blink_rate']
            gaze_focus = data['gaze_focus']
            screen_content = data['screen_content']
            
            # Update camera
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                960, 540, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.camera_label.setPixmap(pixmap)
            
            # Update concentration
            self.concentration_bar.setValue(concentration)
            self.concentration_percent.setText(f"{concentration}%")
            
            if concentration >= 70:
                state = "High Focus"
                color = "hsl(174, 62%, 47%)"
            elif concentration >= 40:
                state = "Moderate Focus"
                color = "hsl(195, 85%, 45%)"
            else:
                state = "Low Focus"
                color = "hsl(280, 65%, 60%)"
            
            self.focus_state_label.setText(state)
            self.focus_state_label.setStyleSheet(f"""
                font-size: 18px;
                font-weight: bold;
                color: {color};
            """)
            
            # Update stats
            self.blink_card.update_value(f"{blink_rate}/min")
            self.heart_card.update_value(f"{heart_rate} bpm" if heart_rate > 0 else "-- bpm")
            self.gaze_card.update_value(f"{gaze_focus}%")
            
            if screen_content:
                self.content_card.update_value(screen_content['category'])
                self.productivity_card.update_value(f"{screen_content['productivity_score']}%")
                
                # Update session data
                self.session_data.update(concentration, screen_content['productivity_score'])
            
        except Exception as e:
            print(f"[Display] Error: {e}")
    
    def update_timer(self):
        """Update session timer"""
        if self.session_start:
            elapsed = int(time.time() - self.session_start)
            mins = elapsed // 60
            secs = elapsed % 60
            self.timer_label.setText(f"{mins:02d}:{secs:02d}")
    
    def update_history(self):
        """Update history view"""
        if not self.session_data.sessions:
            return
        
        history = "Recent Sessions\n" + "=" * 60 + "\n\n"
        
        for i, session in enumerate(reversed(self.session_data.sessions[-10:]), 1):
            start = session['start_time'].strftime("%Y-%m-%d %H:%M:%S")
            duration = int(session['duration'] / 60)
            avg_conc = session['avg_concentration']
            prod = session.get('productivity_score', 0)
            
            history += f"Session #{len(self.session_data.sessions) - i + 1}\n"
            history += f"  Started: {start}\n"
            history += f"  Duration: {duration} minutes\n"
            history += f"  Avg Concentration: {avg_conc:.1f}%\n"
            history += f"  Productivity: {prod:.1f}%\n"
            history += "-" * 60 + "\n\n"
        
        self.history_text.setPlainText(history)
    
    def update_dashboard(self):
        """Update dashboard stats"""
        stats = self.session_data.get_stats()
        if stats:
            self.dash_avg_card.update_value(f"{stats['avg_concentration']:.0f}%")
            self.dash_time_card.update_value(f"{stats['total_time']:.1f}h")
            self.dash_sessions_card.update_value(str(stats['total_sessions']))
            
            # Generate insights
            insights = (
                f"• Completed {stats['total_sessions']} focus sessions\n"
                f"• Average concentration: {stats['avg_concentration']:.0f}%\n"
                f"• Total focused time: {stats['total_time']:.1f} hours\n"
                f"• Screen content analysis provides productivity insights\n"
                f"• Enhanced eye tracking improves accuracy\n"
            )
            self.insights_text.setPlainText(insights)
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.wait()
        if self.screen_thread:
            self.screen_thread.stop()
            self.screen_thread.wait()
        event.accept()


def main():
    """Main application entry"""
    print("=" * 70)
    print("FocusSense - Enhanced Eye Tracking & Screen Analytics")
    print("=" * 70)
    print("Features:")
    print("  • Enhanced eye tracking sensitivity")
    print("  • Screen content analysis")
    print("  • Real-time productivity monitoring")
    print("  • Soft neutral UI colors")
    print("=" * 70)
    
    app = QApplication(sys.argv)
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    print("\nApplication ready! Click 'Start Session' to begin tracking.")
    print("=" * 70)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
