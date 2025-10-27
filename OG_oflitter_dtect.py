import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import tensorflow as tf
import time
import winsound  # For Windows beep sound (use different library for Mac/Linux)

class EnhancedLitterDetector:
    def __init__(self):
        print("Initializing Enhanced Litter Detector with Optical Flow and Ground Line Detection...")
        # Load pre-trained MobileNetV2 for litter detection
        self.model = MobileNetV2(weights='imagenet')

        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5412
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Define expanded litter categories with more keywords
        self.litter_categories = {
            'plastic_bottle': ['water_bottle', 'bottle', 'pop_bottle', 'beer_bottle', 'wine_bottle', 'vessel', 'container', 'flask', 'canteen'],
            'container': ['cup', 'can', 'container', 'coffee_mug', 'measuring_cup', 'beaker', 'bucket', 'bowl', 'goblet' , 'mug', 'chalice'],
            'bag': ['plastic_bag', 'shopping_bag', 'purse', 'backpack', 'packet', 'package', 'pouch', 'sack', 'tote'],
            'food_wrapper': ['packet', 'wrapper', 'envelope', 'package', 'carton', 'cellophane', 'foil', 'baggie', 'sachet'],
            'cardboard': ['cardboard', 'box', 'carton', 'packet', 'package', 'chest', 'crate', 'paperboard'],
            'plastic': ['plastic', 'polymer', 'synthetic', 'styrofoam', 'disposable'],
            'electronics': ['device', 'electronic', 'battery', 'phone', 'gadget', 'computer', 'circuit']
        }

        self.frame_count = 0
        self.confidence_threshold = 0.05
        self.debug_mode = True

        # Track previous litter detection state to prevent continuous beeping
        self.previous_litter_detected = False

        # Performance tracking
        self.last_detection_time = time.time()
        self.fps = 0

        # Add detection interval to improve performance
        self.litter_detection_interval = 2  # Check every 2 frames
        self.last_detected_items = []
        self.last_all_detections = []
        
        # Temporal smoothing
        self.detection_history = {}  # Track detections over time
        self.history_window = 10     # Consider last 10 frames
        self.min_appearances = 2     # Require at least 2 appearances to confirm
        
        # NEW: Initialize optical flow
        self.prev_gray = None
        self.tracks = []  # List to store tracked points
        self.track_len = 10  # Maximum length of tracks
        
        # NEW: Object tracking variables
        self.tracked_objects = {}  # Dictionary to store tracked objects
        self.next_object_id = 0    # Counter for object IDs
        
        # NEW: Ground line definition (y-coordinate, will be adjustable)
        self.ground_line_y = None  # Will be set based on frame height
        self.ground_line_percent = 0.85  # 85% down the frame by default
        
        # NEW: Variables for littering detection
        self.potential_litter = {}  # Track potential litter objects
        self.littering_detected = False
        
        # NEW: For performance optimization
        self.detection_downscale = 0.5  # Process at half resolution for detection
        self.skip_frames = 1  # Process every other frame for heavy operations

    def enhance_for_lighting_invariance(self, frame):
        """Apply lighting-invariant preprocessing using CLAHE"""
        # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to lightness channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels back
        merged = cv2.merge([cl, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        
        return enhanced

    def normalize_colors(self, frame):
        """Normalize colors to reduce lighting effects"""
        # Convert to float and normalize
        norm_img = np.zeros(frame.shape, dtype=np.float32)
        norm_img = cv2.normalize(frame, norm_img, 0, 255, cv2.NORM_MINMAX)
        
        # Convert back to uint8
        return np.uint8(norm_img)

    def auto_white_balance(self, frame):
        """Apply automatic white balance"""
        # Convert to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Get average of A and B channels (a* and b* in Lab color space)
        l, a, b = cv2.split(lab)
        a_avg = np.average(a)
        b_avg = np.average(b)
        
        # Modify the channels to make neutrals more gray
        a = a - ((a_avg - 128) * (a.astype(float) / 255.0) * 0.7)
        b = b - ((b_avg - 128) * (b.astype(float) / 255.0) * 0.7)
        
        # Make sure values stay in valid range
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        
        # Merge back
        balanced = cv2.merge([l, a, b])
        
        # Convert back to BGR
        return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

    def night_mode_detection(self, frame):
        """Special preprocessing for night/dark conditions"""
        # Extreme brightness enhancement
        brightened = cv2.convertScaleAbs(frame, alpha=3.0, beta=50)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(brightened, None, 10, 10, 7, 21)
        
        # Edge enhancement
        sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, sharpening_kernel)
        
        return sharpened

    def detect_lighting_condition(self, frame):
        """Detect lighting condition and select appropriate processing"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate average brightness
        avg_brightness = np.mean(gray)
        
        # Check lighting condition
        if avg_brightness < 40:
            return "very_dark"
        elif avg_brightness < 80:
            return "dark"
        elif avg_brightness > 200:
            return "very_bright"
        elif avg_brightness > 150:
            return "bright"
        else:
            return "normal"

    def prepare_for_detection(self, frame):
        """Apply comprehensive lighting-invariant preprocessing"""
        # Step 1: Apply white balance
        balanced = self.auto_white_balance(frame)
        
        # Step 2: Normalize colors
        normalized = self.normalize_colors(balanced)
        
        # Step 3: Apply CLAHE enhancement
        enhanced = self.enhance_for_lighting_invariance(normalized)
        
        # Step 4: Create multiple versions of the image
        versions = [
            enhanced,                                                # Enhanced version
            cv2.convertScaleAbs(enhanced, alpha=1.5, beta=30),       # Brightened for dark scenes
            cv2.convertScaleAbs(enhanced, alpha=0.8, beta=-20),      # Darkened for bright scenes
            cv2.GaussianBlur(enhanced, (3, 3), 0)                    # Blurred version
        ]
        
        return versions
        
    def preprocess_image(self, frame):
        """Process image for model input"""
        # Resize image to MobileNetV2 input size
        img = cv2.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        return preprocess_input(img)

    def classify_litter(self, predictions):
        decoded_preds = decode_predictions(predictions, top=20)[0]
        detected_items = []
        all_detections = []

        for _, label, confidence in decoded_preds:
            all_detections.append((label, confidence))

        for _, label, confidence in decoded_preds:
            for litter_type, keywords in self.litter_categories.items():
                if any(keyword in label for keyword in keywords) and confidence > self.confidence_threshold:
                    detected_items.append((litter_type, confidence))
                    break

        return detected_items, all_detections

    def detect_hands(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = self.hands.process(rgb_frame)

        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

            # Add hand detection label
            cv2.putText(
                frame,
                f"Hands detected: {len(results.multi_hand_landmarks)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return frame, len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0, results.multi_hand_landmarks

    def get_hand_regions(self, frame, hand_landmarks):
        """Extract regions around hands for focused detection"""
        regions = []
        h, w, _ = frame.shape
        
        if hand_landmarks:
            for landmarks in hand_landmarks:
                # Calculate bounding box around hand
                x_min = w
                y_min = h
                x_max = 0
                y_max = 0
                
                for landmark in landmarks.landmark:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    x_min = min(x_min, px)
                    y_min = min(y_min, py)
                    x_max = max(x_max, px)
                    y_max = max(y_max, py)
                
                # Expand region by 50% to include items held in hand
                width = x_max - x_min
                height = y_max - y_min
                x_min = max(0, x_min - width//2)
                y_min = max(0, y_min - height//2)
                x_max = min(w, x_max + width//2)
                y_max = min(h, y_max + height//2)
                
                # Extract region
                if x_min < x_max and y_min < y_max:
                    hand_region = frame[y_min:y_max, x_min:x_max]
                    if hand_region.size > 0:
                        regions.append((hand_region, (x_min, y_min, x_max, y_max)))
        
        return regions

    def update_detection_history(self, detected_items):
        """Update detection history for temporal smoothing"""
        current_frame = self.frame_count
        
        # Add new detections to history
        for item_type, confidence in detected_items:
            if item_type not in self.detection_history:
                self.detection_history[item_type] = []
            self.detection_history[item_type].append((current_frame, confidence))
        
        # Remove old detections
        for item_type in list(self.detection_history.keys()):
            self.detection_history[item_type] = [
                (frame, conf) for frame, conf in self.detection_history[item_type]
                if current_frame - frame < self.history_window
            ]
            
            # Clean up empty lists
            if not self.detection_history[item_type]:
                del self.detection_history[item_type]

    def get_smoothed_detections(self):
        """Get temporally smoothed detections"""
        smoothed_items = []
        
        for item_type, detections in self.detection_history.items():
            if len(detections) >= self.min_appearances:
                # Calculate average confidence
                avg_confidence = sum(conf for _, conf in detections) / len(detections)
                smoothed_items.append((item_type, avg_confidence))
        
        return smoothed_items

    def play_beep(self, litter_detected=True):
        """Play beep sound if litter is detected"""
        try:
            # For Windows - frequency 800Hz, duration 200ms
            winsound.Beep(1000, 300)  # Higher pitch for littering alert
        except:
            # For other platforms, print a message
            print('\a')  # ASCII bell character

    # NEW: Optical flow methods
    def calculate_optical_flow(self, frame):
        """Calculate optical flow between consecutive frames"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            # Initialize tracking points - use good features to track
            p = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, 
                                       qualityLevel=0.3, minDistance=7, 
                                       blockSize=7)
            if p is not None:
                for x, y in p.reshape(-1, 2):
                    self.tracks.append([(x, y)])
            return frame, []
        
        # Calculate optical flow
        if len(self.tracks) > 0:
            p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, p0, None,
                winSize=(15, 15), maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            
            # Get back good tracks
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            
            flow_vectors = []
            # Add new tracking points periodically
            if self.frame_count % 5 == 0:
                mask = np.zeros_like(gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                
                p = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100,
                                          qualityLevel=0.3, minDistance=7,
                                          blockSize=7)
                if p is not None:
                    for x, y in p.reshape(-1, 2):
                        self.tracks.append([(x, y)])
            
            # Now update the previous points
            new_tracks = []
            for i, (tr, (x, y), (old_x, old_y)) in enumerate(zip(self.tracks, good_new, good_old)):
                tr.append((x, y))
                if len(tr) > self.track_len:
                    tr = tr[-self.track_len:]
                new_tracks.append(tr)
                
                # Draw the tracks
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                if len(tr) > 1:
                    cv2.polylines(frame, [np.int32(tr)], False, (0, 255, 0), 1)
                
                # Calculate flow vector
                flow_vectors.append({
                    'position': (x, y),
                    'velocity': (x - old_x, y - old_y),
                    'track_id': i
                })
                
            self.tracks = new_tracks
            
            # Update previous gray image
            self.prev_gray = gray.copy()
            
            return frame, flow_vectors
        else:
            # If no tracks, reinitialize
            p = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100,
                                      qualityLevel=0.3, minDistance=7,
                                      blockSize=7)
            if p is not None:
                for x, y in p.reshape(-1, 2):
                    self.tracks.append([(x, y)])
            
            self.prev_gray = gray.copy()
            return frame, []

    # NEW: Object detection and tracking methods
    def detect_objects(self, frame, hand_landmarks):
        """Detect potential litter objects in the frame"""
        # We'll use a combination of our existing litter detection
        # and motion tracking to identify potential objects
        
        objects = []
        
        # First, get regions around hands which might contain objects
        hand_regions = self.get_hand_regions(frame, hand_landmarks)
        
        for hand_region, bbox in hand_regions:
            if hand_region.size > 0:
                # Preprocess and predict litter in hand region
                preprocessed = self.preprocess_image(hand_region)
                predictions = self.model.predict(preprocessed, verbose=0)
                region_items, _ = self.classify_litter(predictions)
                
                # If we detected potential litter items, create an object
                for item_type, confidence in region_items:
                    # Calculate center of the bbox
                    x_min, y_min, x_max, y_max = bbox
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    objects.append({
                        'type': item_type,
                        'confidence': confidence,
                        'bbox': bbox,
                        'center': (center_x, center_y),
                        'size': (width, height),
                        'in_hand': True,
                        'velocity': (0, 0)  # Will be updated with optical flow
                    })
        
        return objects

    def update_tracked_objects(self, detected_objects, flow_vectors):
        """Update tracking information for detected objects"""
        # First update velocity of detected objects using optical flow data
        for obj in detected_objects:
            center_x, center_y = obj['center']
            
            # Find closest flow vectors to update object velocity
            closest_vectors = []
            for vector in flow_vectors:
                x, y = vector['position']
                # Check if vector is within object bounding box
                x_min, y_min, x_max, y_max = obj['bbox']
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    closest_vectors.append(vector)
            
            # Calculate average velocity from flow vectors
            if closest_vectors:
                avg_vx = sum(v['velocity'][0] for v in closest_vectors) / len(closest_vectors)
                avg_vy = sum(v['velocity'][1] for v in closest_vectors) / len(closest_vectors)
                obj['velocity'] = (avg_vx, avg_vy)
        
        # Match detected objects with existing tracked objects
        if not self.tracked_objects:
            # First frame, initialize all objects
            for obj in detected_objects:
                self.tracked_objects[self.next_object_id] = {
                    'type': obj['type'],
                    'confidence': obj['confidence'],
                    'bbox': obj['bbox'],
                    'center': obj['center'],
                    'size': obj['size'],
                    'velocity': obj['velocity'],
                    'in_hand': obj['in_hand'],
                    'history': [obj['center']],
                    'last_seen': self.frame_count,
                    'falling': False,
                    'crossed_ground_line': False
                }
                self.next_object_id += 1
        else:
            # Update existing or add new objects
            matched_ids = set()
            
            for obj in detected_objects:
                best_match = None
                best_distance = float('inf')
                
                # Find closest tracked object
                for obj_id, tracked_obj in self.tracked_objects.items():
                    if self.frame_count - tracked_obj['last_seen'] > 30:
                        # Skip objects not seen recently
                        continue
                    
                    # Calculate distance between centers
                    tx, ty = tracked_obj['center']
                    ox, oy = obj['center']
                    distance = np.sqrt((tx - ox)**2 + (ty - oy)**2)
                    
                    # Simple matching heuristic - closest with same type
                    if distance < best_distance and distance < 100:  # Maximum distance threshold
                        best_distance = distance
                        best_match = obj_id
                
                if best_match is not None:
                    # Update existing object
                    self.tracked_objects[best_match].update({
                        'bbox': obj['bbox'],
                        'center': obj['center'],
                        'size': obj['size'],
                        'confidence': max(obj['confidence'], self.tracked_objects[best_match]['confidence']),
                        'velocity': obj['velocity'],
                        'in_hand': obj['in_hand'],
                        'last_seen': self.frame_count
                    })
                    self.tracked_objects[best_match]['history'].append(obj['center'])
                    if len(self.tracked_objects[best_match]['history']) > 30:
                        self.tracked_objects[best_match]['history'] = self.tracked_objects[best_match]['history'][-30:]
                    
                    matched_ids.add(best_match)
                else:
                    # Add new object
                    self.tracked_objects[self.next_object_id] = {
                        'type': obj['type'],
                        'confidence': obj['confidence'],
                        'bbox': obj['bbox'],
                        'center': obj['center'],
                        'size': obj['size'],
                        'velocity': obj['velocity'],
                        'in_hand': obj['in_hand'],
                        'history': [obj['center']],
                        'last_seen': self.frame_count,
                        'falling': False,
                        'crossed_ground_line': False
                    }
                    self.next_object_id += 1
            
            # Update objects not found in this frame
            for obj_id in list(self.tracked_objects.keys()):
                if obj_id not in matched_ids:
                    # Object not seen in this frame
                    if self.frame_count - self.tracked_objects[obj_id]['last_seen'] > 30:
                        # Remove if not seen for too long
                        del self.tracked_objects[obj_id]
                    else:
                        # Propagate position based on velocity
                        vx, vy = self.tracked_objects[obj_id]['velocity']
                        cx, cy = self.tracked_objects[obj_id]['center']
                        new_cx = cx + vx
                        new_cy = cy + vy
                        self.tracked_objects[obj_id]['center'] = (new_cx, new_cy)
                        self.tracked_objects[obj_id]['history'].append((new_cx, new_cy))
                        if len(self.tracked_objects[obj_id]['history']) > 30:
                            self.tracked_objects[obj_id]['history'] = self.tracked_objects[obj_id]['history'][-30:]
                        
                        # Update velocity - add gravitational acceleration for falling objects
                        if not self.tracked_objects[obj_id]['in_hand'] and vy > 0:
                            self.tracked_objects[obj_id]['falling'] = True
                            self.tracked_objects[obj_id]['velocity'] = (vx * 0.95, vy * 1.05)  # Simulate gravity
        
        return self.tracked_objects

    def detect_littering(self, frame_height):
        """Detect littering events based on object trajectories and ground line"""
        littering_detected = False
        littered_objects = []
        
        # Ensure ground line is set
        if self.ground_line_y is None:
            self.ground_line_y = int(frame_height * self.ground_line_percent)
        
        for obj_id, obj in self.tracked_objects.items():
            # Skip objects already processed
            if obj['crossed_ground_line']:
                continue
                
            # Check if object was in hand and is now falling
            cx, cy = obj['center']
            vx, vy = obj['velocity']
            
            # Define littering conditions:
            # 1. Object was previously in hand
            # 2. Object is falling (positive y velocity)
            # 3. Object has crossed ground line
            
            history_len = len(obj['history'])
            was_in_hand = False
            
            # Check if object was in hand at some point
            if obj['in_hand'] or (history_len > 5 and any(y < self.ground_line_y * 0.7 for _, y in obj['history'][:history_len//2])):
                was_in_hand = True
            
            # Check if object is below ground line
            if cy > self.ground_line_y and was_in_hand and vy > 0:
                obj['crossed_ground_line'] = True
                littering_detected = True
                littered_objects.append(obj_id)
        
        return littering_detected, littered_objects

    def draw_tracked_objects(self, frame):
        """Draw tracked objects and their trajectories"""
        for obj_id, obj in self.tracked_objects.items():
            if self.frame_count - obj['last_seen'] > 5:
                continue  # Skip drawing objects not seen recently
                
            # Draw bounding box
            x_min, y_min, x_max, y_max = obj['bbox']
            color = (0, 0, 255) if obj['crossed_ground_line'] else (255, 0, 0)
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            
            # Draw ID and type
            cv2.putText(frame, 
                      f"ID:{obj_id} {obj['type']}", 
                      (int(x_min), int(y_min) - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.5, 
                      color, 
                      2)
            
            # Draw trajectory
            if len(obj['history']) > 1:
                # Convert history points to numpy array for polylines
                trajectory = np.array(obj['history'], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [trajectory], False, color, 2)
            
            # Draw velocity vector
            cx, cy = obj['center']
            vx, vy = obj['velocity']
            cv2.arrowedLine(frame, 
                          (int(cx), int(cy)), 
                          (int(cx + vx*3), int(cy + vy*3)), 
                          (0, 255, 255), 
                          2)
        
        return frame

    def draw_ground_line(self, frame):
        """Draw the ground line on the frame"""
        if self.ground_line_y is None:
            self.ground_line_y = int(frame.shape[0] * self.ground_line_percent)
            
        cv2.line(frame, 
               (0, self.ground_line_y), 
               (frame.shape[1], self.ground_line_y), 
               (255, 0, 0), 
               2)
        
        cv2.putText(frame, 
                  "Ground Line", 
                  (10, self.ground_line_y - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 
                  0.7, 
                  (255, 0, 0), 
                  2)
        
        return frame

    def process_frame(self, frame):
        """Process a single frame with enhanced detection and tracking"""
        self.frame_count += 1
        start_time = time.time()

        if frame is None:
            print("Received empty frame")
            return None

        # Resize for faster processing if needed
        if self.detection_downscale < 1.0:
            frame_small = cv2.resize(frame, (0, 0), fx=self.detection_downscale, fy=self.detection_downscale)
        else:
            frame_small = frame.copy()

        # Create status area
        status_area = np.zeros((300, frame.shape[1], 3), dtype=np.uint8)

        try:
            # Set ground line if not set
            if self.ground_line_y is None:
                self.ground_line_y = int(frame.shape[0] * self.ground_line_percent)
            
            # Process frame only on certain intervals for heavy operations
            if self.frame_count % (self.skip_frames + 1) == 0:
                # Detect lighting condition
                lighting = self.detect_lighting_condition(frame_small)
                
                # Apply appropriate preprocessing based on lighting
                if lighting == "very_dark":
                    processed_frame = self.night_mode_detection(frame_small)
                else:
                    # Get multiple versions of the image for different lighting conditions
                    frame_versions = self.prepare_for_detection(frame_small)
                    processed_frame = frame_versions[0]  # Use the enhanced version for display
            else:
                # Use previous processing results for in-between frames
                processed_frame = frame_small.copy()
                lighting = "skipped"
            
            # Always detect hands in every frame (critical for tracking)
            frame_with_hands, num_hands, hand_landmarks = self.detect_hands(frame.copy())
            
            # Draw ground line
            frame_with_hands = self.draw_ground_line(frame_with_hands)
            
            # NEW: Calculate optical flow for object
            # NEW: Calculate optical flow for object tracking
            flow_frame, flow_vectors = self.calculate_optical_flow(frame)
            
            # Detect potential litter objects
            detected_objects = self.detect_objects(frame, hand_landmarks)
            
            # Update tracked objects with new detections
            tracked_objects = self.update_tracked_objects(detected_objects, flow_vectors)
            
            # Detect littering events
            littering_detected, littered_objects = self.detect_littering(frame.shape[0])
            
            # Alert if littering is detected
            if littering_detected and not self.littering_detected:
                self.littering_detected = True
                self.play_beep(litter_detected=True)
                print("ALERT: Potential littering detected!")
            elif not littering_detected:
                self.littering_detected = False
            
            # Draw tracked objects and trajectories
            result_frame = self.draw_tracked_objects(frame_with_hands)
            
            # Run litter detection on some frames
            if self.frame_count % self.litter_detection_interval == 0:
                if num_hands > 0:
                    # Focus detection on hand regions when hands are visible
                    hand_regions = self.get_hand_regions(frame, hand_landmarks)
                    detected_items = []
                    
                    for hand_region, _ in hand_regions:
                        if hand_region.size > 0:  # Ensure region is not empty
                            # Preprocess and predict
                            preprocessed = self.preprocess_image(hand_region)
                            predictions = self.model.predict(preprocessed, verbose=0)
                            region_items, _ = self.classify_litter(predictions)
                            detected_items.extend(region_items)
                else:
                    # Process whole frame when no hands detected
                    # Select best processed frame based on lighting condition
                    preprocessed = self.preprocess_image(processed_frame)
                    predictions = self.model.predict(preprocessed, verbose=0)
                    detected_items, all_detections = self.classify_litter(predictions)
                
                # Update temporal detection history
                self.update_detection_history(detected_items)
                
                # Get smoothed detections
                smoothed_items = self.get_smoothed_detections()
                
                # Update for alert logic
                self.last_detected_items = smoothed_items
                
                # Sound alert if litter detected (and wasn't previously detected)
                litter_detected = len(smoothed_items) > 0
                if litter_detected and not self.previous_litter_detected:
                    self.play_beep()
                self.previous_litter_detected = litter_detected
            
            # Calculate FPS
            end_time = time.time()
            processing_time = end_time - start_time
            self.fps = 1.0 / processing_time if processing_time > 0 else 0
            
            # Show FPS and detection info in status area
            status_area.fill(0)  # Clear status area
            
            # Draw status information
            cv2.putText(status_area, f"FPS: {self.fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_area, f"Lighting: {lighting}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_area, f"Frame: {self.frame_count}", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_area, f"Hands: {num_hands}", (10, 120), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(status_area, f"Tracked objects: {len(self.tracked_objects)}", (10, 150), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show detected litter items
            y_offset = 180
            cv2.putText(status_area, "Detected Items:", (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            
            for i, (item_type, confidence) in enumerate(self.last_detected_items):
                if i >= 5:  # Show max 5 items to avoid overflow
                    cv2.putText(status_area, "...", (10, y_offset), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
                    
                cv2.putText(status_area, f"{item_type}: {confidence:.2f}", (10, y_offset), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Show littering status
            if self.littering_detected:
                cv2.putText(status_area, "LITTERING DETECTED!", (frame.shape[1]//2 - 150, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Combine frame and status
            result = np.vstack([result_frame, status_area])
            
            return result
        
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            # Return original frame with error message
            cv2.putText(frame, f"Processing error: {str(e)}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

def main():
    """Main function to run the litter detector"""
    # Initialize detector
    detector = EnhancedLitterDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    print("Starting Enhanced Litter Detection...")
    print("Press 'q' to quit, 'g' to adjust ground line position")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
            break
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Show result
        cv2.imshow("Enhanced Litter Detector", result)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            # Quit
            break
        elif key == ord('g'):
            # Adjust ground line
            height = frame.shape[0]
            current_percent = detector.ground_line_percent
            new_percent = current_percent + 0.05
            if new_percent > 0.95:
                new_percent = 0.5  # Reset to middle
            
            detector.ground_line_percent = new_percent
            detector.ground_line_y = int(height * new_percent)
            print(f"Ground line adjusted to {new_percent*100:.0f}% of frame height")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()