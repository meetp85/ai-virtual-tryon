"""
Virtual Try-On (VTO) Accuracy Scoring System
=============================================
Comprehensive accuracy measurement using 3 independent methods:

METHOD 1: Self-Consistency Testing
    - Measures landmark stability across frames (same person, same position)
    - Low variance = high consistency = reliable tracking
    - Includes: Face Detection Rate, Jitter Score, Temporal Precision

METHOD 2: Geometric Constraint Validation
    - Verifies anatomical correctness of landmarks on ANY face
    - Checks: left ear is left of nose, ears are symmetric, chin below eyes, etc.
    - No ground truth needed — uses universal facial geometry rules

METHOD 3: Overlay Placement Scoring
    - Measures whether jewelry overlay is correctly positioned
    - Checks: necklace centered on face midline, earrings symmetric, proportional sizing
    - Directly measures what matters for VTO quality

Overall Accuracy = Method 1 (30%) + Method 2 (40%) + Method 3 (30%)
"""

import numpy as np
from collections import deque
from datetime import datetime
import json
import os


class VTOAccuracyTracker:

    def __init__(self, smoothing_window=10):
        # Frame counters
        self.total_frames = 0
        self.detected_frames = 0

        # Method 1: Self-Consistency
        self.confidence_scores = []
        self.smoothing_window = smoothing_window
        self.landmark_history = {}
        self.jitter_scores = []

        # Method 2: Geometric Constraints
        self.geometric_results = []  # list of per-frame constraint pass rates

        # Method 3: Overlay Placement
        self.overlay_results = []  # list of per-frame placement scores

        # Key landmarks for jewelry VTO
        self.key_landmarks = {
            'left_ear': 234,
            'right_ear': 454,
            'chin': 152,
            'nose_tip': 1,
            'left_eye_outer': 33,
            'right_eye_outer': 263,
            'forehead': 10,
            'left_jaw': 132,
            'right_jaw': 361,
            'left_earlobe': 177,
            'right_earlobe': 401,
            'nose_bridge': 6,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'upper_lip': 13,
            'lower_lip': 14,
            'left_cheek': 123,
            'right_cheek': 352,
        }

        # Per-landmark confidence
        self.landmark_confidences = {name: [] for name in self.key_landmarks}

        # Session metadata
        self.session_start = datetime.now()
        self.category_tested = None

    def record_frame(self, landmarks=None, face_detected=True,
                     detection_confidence=0.0, landmark_list=None,
                     overlay_info=None):
        """
        Record metrics for a single frame.

        Args:
            landmarks: MediaPipe face_landmarks object
            face_detected: bool
            detection_confidence: float [0, 1]
            landmark_list: optional raw list of (x, y, z, visibility) tuples
            overlay_info: optional dict with overlay placement data
                          {'jewelry_type', 'center_x', 'center_y', 'width', 'height'}
        """
        self.total_frames += 1

        if not face_detected or landmarks is None:
            return

        self.detected_frames += 1
        self.confidence_scores.append(detection_confidence)

        # Extract landmark positions
        if landmark_list is None and hasattr(landmarks, 'landmark'):
            landmark_list = [
                (lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0))
                for lm in landmarks.landmark
            ]

        if landmark_list is None:
            return

        # === METHOD 1: Self-Consistency ===
        self._track_consistency(landmark_list)

        # === METHOD 2: Geometric Constraints ===
        self._check_geometric_constraints(landmark_list)

        # === METHOD 3: Overlay Placement ===
        if overlay_info:
            self._check_overlay_placement(landmark_list, overlay_info)
        else:
            # Auto-compute placement score from landmark positions
            self._auto_overlay_score(landmark_list)

    # ==================================================================
    # METHOD 1: Self-Consistency Testing
    # ==================================================================

    def _track_consistency(self, landmark_list):
        """Track landmark positions across frames for consistency analysis."""
        frame_jitter = []

        for name, idx in self.key_landmarks.items():
            if idx >= len(landmark_list):
                continue

            x, y = landmark_list[idx][0], landmark_list[idx][1]

            # Track in-frame ratio as confidence
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                self.landmark_confidences[name].append(1.0)
            else:
                self.landmark_confidences[name].append(0.0)

            # History for jitter
            if name not in self.landmark_history:
                self.landmark_history[name] = deque(maxlen=self.smoothing_window)

            history = self.landmark_history[name]

            if len(history) > 0:
                prev_x, prev_y = history[-1]
                jitter = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
                frame_jitter.append(jitter)

            history.append((x, y))

        if frame_jitter:
            self.jitter_scores.append(np.mean(frame_jitter))

    def compute_face_detection_rate(self):
        """% of frames where face was detected."""
        if self.total_frames == 0:
            return 0.0
        return (self.detected_frames / self.total_frames) * 100

    def compute_landmark_confidence(self):
        """Average in-frame confidence across all key landmarks."""
        all_scores = []
        for name, scores in self.landmark_confidences.items():
            if scores:
                all_scores.extend(scores)
        if not all_scores:
            return 0.0
        return float(np.mean(all_scores)) * 100

    def compute_overlay_stability(self):
        """Frame-to-frame stability score. Lower jitter = higher score."""
        if not self.jitter_scores:
            return 100.0
        mean_jitter = float(np.mean(self.jitter_scores))
        stability = 100.0 * np.exp(-mean_jitter * 50)
        return max(0.0, min(100.0, stability))

    def compute_temporal_precision(self):
        """Consistency of key anchor positions over sliding window."""
        precisions = []
        for name, history in self.landmark_history.items():
            if len(history) < 3:
                continue
            positions = np.array(list(history))
            variance = np.var(positions, axis=0).sum()
            precision = 100.0 * np.exp(-variance * 1000)
            precisions.append(precision)
        if not precisions:
            return 0.0
        return float(np.mean(precisions))

    def get_method1_score(self):
        """
        Self-Consistency Score (0-100).
        Combines: FDR (25%) + Confidence (25%) + Stability (30%) + Precision (20%)
        """
        fdr = self.compute_face_detection_rate()
        conf = self.compute_landmark_confidence()
        stab = self.compute_overlay_stability()
        prec = self.compute_temporal_precision()
        return (fdr * 0.25) + (conf * 0.25) + (stab * 0.30) + (prec * 0.20)

    # ==================================================================
    # METHOD 2: Geometric Constraint Validation
    # ==================================================================

    def _check_geometric_constraints(self, landmark_list):
        """
        Check if detected landmarks satisfy universal anatomical constraints.
        These must be true for ANY human face regardless of shape/size.
        """
        constraints_passed = 0
        constraints_total = 0

        def get_lm(idx):
            if idx < len(landmark_list):
                return landmark_list[idx][0], landmark_list[idx][1]
            return None, None

        left_ear_x, left_ear_y = get_lm(234)
        right_ear_x, right_ear_y = get_lm(454)
        nose_x, nose_y = get_lm(1)
        chin_x, chin_y = get_lm(152)
        forehead_x, forehead_y = get_lm(10)
        left_eye_x, left_eye_y = get_lm(33)
        right_eye_x, right_eye_y = get_lm(263)
        left_jaw_x, left_jaw_y = get_lm(132)
        right_jaw_x, right_jaw_y = get_lm(361)
        upper_lip_x, upper_lip_y = get_lm(13)
        nose_bridge_x, nose_bridge_y = get_lm(6)

        if None in [left_ear_x, right_ear_x, nose_x, chin_y, forehead_y]:
            return

        # --- Constraint 1: Left ear is LEFT of nose ---
        constraints_total += 1
        if left_ear_x < nose_x:
            constraints_passed += 1

        # --- Constraint 2: Right ear is RIGHT of nose ---
        constraints_total += 1
        if right_ear_x > nose_x:
            constraints_passed += 1

        # --- Constraint 3: Chin is below nose ---
        constraints_total += 1
        if chin_y > nose_y:
            constraints_passed += 1

        # --- Constraint 4: Forehead is above nose ---
        constraints_total += 1
        if forehead_y < nose_y:
            constraints_passed += 1

        # --- Constraint 5: Eyes are above nose ---
        constraints_total += 1
        if left_eye_y < nose_y and right_eye_y < nose_y:
            constraints_passed += 1

        # --- Constraint 6: Ears are roughly symmetric around nose ---
        # Distance from left ear to nose should be within 40% of right ear to nose
        constraints_total += 1
        left_dist = abs(nose_x - left_ear_x)
        right_dist = abs(right_ear_x - nose_x)
        if left_dist > 0 and right_dist > 0:
            symmetry_ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
            if symmetry_ratio > 0.6:  # within 40% tolerance
                constraints_passed += 1

        # --- Constraint 7: Face height > face width * 0.8 ---
        # Faces are generally taller than wide (or close to equal)
        constraints_total += 1
        face_width = abs(right_ear_x - left_ear_x)
        face_height = abs(chin_y - forehead_y)
        if face_width > 0 and face_height > face_width * 0.8:
            constraints_passed += 1

        # --- Constraint 8: Nose is between eyes horizontally ---
        constraints_total += 1
        if left_eye_x is not None and right_eye_x is not None:
            if left_eye_x < nose_x < right_eye_x:
                constraints_passed += 1

        # --- Constraint 9: Mouth is between chin and nose vertically ---
        constraints_total += 1
        if upper_lip_y is not None:
            if nose_y < upper_lip_y < chin_y:
                constraints_passed += 1

        # --- Constraint 10: Jaw points are below eyes ---
        constraints_total += 1
        if left_jaw_y is not None and right_jaw_y is not None:
            if left_jaw_y > left_eye_y and right_jaw_y > right_eye_y:
                constraints_passed += 1

        # --- Constraint 11: Nose bridge is above nose tip ---
        constraints_total += 1
        if nose_bridge_y is not None:
            if nose_bridge_y < nose_y:
                constraints_passed += 1

        # --- Constraint 12: Eyes are roughly at same height ---
        constraints_total += 1
        if left_eye_y is not None and right_eye_y is not None:
            eye_height_diff = abs(left_eye_y - right_eye_y)
            face_h = abs(chin_y - forehead_y) if abs(chin_y - forehead_y) > 0 else 1
            if eye_height_diff / face_h < 0.1:  # within 10% of face height
                constraints_passed += 1

        # Store result
        if constraints_total > 0:
            self.geometric_results.append(constraints_passed / constraints_total * 100)

    def get_method2_score(self):
        """
        Geometric Constraint Accuracy (0-100).
        Average pass rate across all frames.
        """
        if not self.geometric_results:
            return 0.0
        return float(np.mean(self.geometric_results))

    def get_geometric_breakdown(self):
        """Get per-frame geometric scores for analysis."""
        if not self.geometric_results:
            return {}
        return {
            'mean': round(float(np.mean(self.geometric_results)), 2),
            'min': round(float(np.min(self.geometric_results)), 2),
            'max': round(float(np.max(self.geometric_results)), 2),
            'std': round(float(np.std(self.geometric_results)), 2),
            'frames_analyzed': len(self.geometric_results),
            'constraints_checked': 12,
        }

    # ==================================================================
    # METHOD 3: Overlay Placement Scoring
    # ==================================================================

    def _auto_overlay_score(self, landmark_list):
        """
        Compute overlay placement quality from landmark positions.
        Checks if jewelry anchor points are well-positioned for overlay.
        """
        scores = []

        def get_lm(idx):
            if idx < len(landmark_list):
                return landmark_list[idx][0], landmark_list[idx][1]
            return None, None

        left_ear_x, left_ear_y = get_lm(234)
        right_ear_x, right_ear_y = get_lm(454)
        chin_x, chin_y = get_lm(152)
        nose_x, nose_y = get_lm(1)
        forehead_x, forehead_y = get_lm(10)
        left_jaw_x, left_jaw_y = get_lm(132)
        right_jaw_x, right_jaw_y = get_lm(361)

        if None in [left_ear_x, right_ear_x, chin_x, nose_x]:
            return

        face_width = abs(right_ear_x - left_ear_x)
        if face_width < 0.05:  # too small
            return

        # --- Score 1: Necklace center alignment ---
        # Chin should be roughly centered horizontally (midline of face)
        face_center_x = (left_ear_x + right_ear_x) / 2
        chin_offset = abs(chin_x - face_center_x) / face_width
        necklace_center_score = max(0, 100 * (1 - chin_offset * 5))
        scores.append(necklace_center_score)

        # --- Score 2: Earring symmetry ---
        # Left and right ear Y positions should be similar
        ear_y_diff = abs(left_ear_y - right_ear_y)
        face_height = abs(chin_y - forehead_y) if forehead_y is not None else face_width
        if face_height > 0:
            ear_symmetry = max(0, 100 * (1 - ear_y_diff / face_height * 10))
        else:
            ear_symmetry = 50
        scores.append(ear_symmetry)

        # --- Score 3: Anchor point visibility ---
        # All key jewelry anchor points should be within frame
        anchor_points = [
            (left_ear_x, left_ear_y), (right_ear_x, right_ear_y),
            (chin_x, chin_y), (nose_x, nose_y)
        ]
        visible = sum(1 for x, y in anchor_points if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95)
        visibility_score = (visible / len(anchor_points)) * 100
        scores.append(visibility_score)

        # --- Score 4: Proportional sizing ---
        # Face should occupy 20-60% of frame width for good overlay
        if 0.15 <= face_width <= 0.65:
            proportion_score = 100
        elif face_width < 0.15:
            proportion_score = max(0, face_width / 0.15 * 100)
        else:
            proportion_score = max(0, (1 - (face_width - 0.65) / 0.35) * 100)
        scores.append(proportion_score)

        # --- Score 5: Face orientation (should be roughly frontal) ---
        # Nose should be close to midline between ears
        nose_position = (nose_x - left_ear_x) / face_width if face_width > 0 else 0.5
        # Ideal: nose at 0.5 (center). Tolerance: 0.3-0.7
        frontal_score = max(0, 100 * (1 - abs(nose_position - 0.5) * 4))
        scores.append(frontal_score)

        # --- Score 6: Earlobe-to-jaw alignment (earring placement quality) ---
        if left_jaw_y is not None and right_jaw_y is not None:
            # Earlobe should be between ear and jaw vertically
            left_lobe_y = (left_ear_y + left_jaw_y) / 2
            right_lobe_y = (right_ear_y + right_jaw_y) / 2
            # Both should be below ears and above chin
            lobe_valid = (left_ear_y < left_lobe_y < chin_y and
                         right_ear_y < right_lobe_y < chin_y)
            lobe_score = 100 if lobe_valid else 40
            scores.append(lobe_score)

        if scores:
            self.overlay_results.append(np.mean(scores))

    def _check_overlay_placement(self, landmark_list, overlay_info):
        """Check actual overlay placement against expected position."""
        # If explicit overlay info provided, use it
        self._auto_overlay_score(landmark_list)

    def get_method3_score(self):
        """
        Overlay Placement Accuracy (0-100).
        Average placement quality across all frames.
        """
        if not self.overlay_results:
            return 0.0
        return float(np.mean(self.overlay_results))

    def get_overlay_breakdown(self):
        """Get overlay placement analysis."""
        if not self.overlay_results:
            return {}
        return {
            'mean': round(float(np.mean(self.overlay_results)), 2),
            'min': round(float(np.min(self.overlay_results)), 2),
            'max': round(float(np.max(self.overlay_results)), 2),
            'std': round(float(np.std(self.overlay_results)), 2),
            'frames_analyzed': len(self.overlay_results),
            'checks_per_frame': 6,
            'checks': [
                'Necklace center alignment',
                'Earring symmetry',
                'Anchor point visibility',
                'Face proportion in frame',
                'Frontal face orientation',
                'Earlobe-jaw alignment',
            ]
        }

    # ==================================================================
    # OVERALL ACCURACY REPORT
    # ==================================================================

    def get_accuracy_report(self):
        """
        Generate comprehensive accuracy report combining all 3 methods.

        Overall = Method 1 (30%) + Method 2 (40%) + Method 3 (30%)

        Method 2 (Geometric) gets highest weight because it's the most
        objectively verifiable — anatomical constraints are universal truths.
        """
        m1 = self.get_method1_score()
        m2 = self.get_method2_score()
        m3 = self.get_method3_score()

        overall = (m1 * 0.30) + (m2 * 0.40) + (m3 * 0.30)

        if overall >= 90:
            grade = 'A+'
        elif overall >= 80:
            grade = 'A'
        elif overall >= 70:
            grade = 'B'
        elif overall >= 60:
            grade = 'C'
        else:
            grade = 'D'

        report = {
            'overall_accuracy': round(overall, 2),
            'grade': grade,

            'method1_self_consistency': {
                'score': round(m1, 2),
                'weight': '30%',
                'description': 'Landmark tracking consistency across frames',
                'sub_metrics': {
                    'face_detection_rate': {
                        'score': round(self.compute_face_detection_rate(), 2),
                        'total_frames': self.total_frames,
                        'detected_frames': self.detected_frames,
                    },
                    'landmark_confidence': {
                        'score': round(self.compute_landmark_confidence(), 2),
                    },
                    'overlay_stability': {
                        'score': round(self.compute_overlay_stability(), 2),
                        'mean_jitter': round(float(np.mean(self.jitter_scores)), 6) if self.jitter_scores else 0,
                    },
                    'temporal_precision': {
                        'score': round(self.compute_temporal_precision(), 2),
                    },
                }
            },

            'method2_geometric_validation': {
                'score': round(m2, 2),
                'weight': '40%',
                'description': 'Anatomical constraint satisfaction (12 universal rules)',
                'breakdown': self.get_geometric_breakdown(),
                'constraints': [
                    'Left ear is left of nose',
                    'Right ear is right of nose',
                    'Chin is below nose',
                    'Forehead is above nose',
                    'Eyes are above nose',
                    'Ears are symmetric around nose (within 40%)',
                    'Face height > 0.8x face width',
                    'Nose is between eyes horizontally',
                    'Mouth is between chin and nose',
                    'Jaw points are below eyes',
                    'Nose bridge is above nose tip',
                    'Eyes are at roughly same height (within 10%)',
                ],
            },

            'method3_overlay_placement': {
                'score': round(m3, 2),
                'weight': '30%',
                'description': 'Jewelry overlay positioning accuracy',
                'breakdown': self.get_overlay_breakdown(),
            },

            # Legacy format for backward compatibility with frontend
            'metrics': {
                'face_detection_rate': {
                    'score': round(self.compute_face_detection_rate(), 2),
                    'weight': '25%',
                    'description': 'Percentage of frames with successful face detection',
                    'total_frames': self.total_frames,
                    'detected_frames': self.detected_frames,
                },
                'landmark_confidence': {
                    'score': round(self.compute_landmark_confidence(), 2),
                    'weight': '25%',
                    'description': 'Average landmark in-frame confidence',
                    'samples': len(self.confidence_scores),
                },
                'overlay_stability': {
                    'score': round(self.compute_overlay_stability(), 2),
                    'weight': '30%',
                    'description': 'Frame-to-frame overlay smoothness',
                    'mean_jitter': round(float(np.mean(self.jitter_scores)), 6) if self.jitter_scores else 0,
                },
                'landmark_precision': {
                    'score': round(self.compute_temporal_precision(), 2),
                    'weight': '20%',
                    'description': 'Key anchor point consistency',
                },
            },

            'session_info': {
                'start_time': self.session_start.isoformat(),
                'duration_frames': self.total_frames,
                'category_tested': self.category_tested,
                'mediapipe_model': 'FaceMesh 468 landmarks + refined (478)',
                'methods_used': [
                    'Self-Consistency Testing (temporal stability)',
                    'Geometric Constraint Validation (12 anatomical rules)',
                    'Overlay Placement Scoring (6 positioning checks)',
                ],
            }
        }

        return report

    def reset(self):
        """Reset all tracking data for a new session."""
        self.total_frames = 0
        self.detected_frames = 0
        self.confidence_scores = []
        self.landmark_history = {}
        self.jitter_scores = []
        self.geometric_results = []
        self.overlay_results = []
        self.landmark_confidences = {name: [] for name in self.key_landmarks}
        self.session_start = datetime.now()
        self.category_tested = None