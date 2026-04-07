import time

import cv2
import numpy as np

from .types import FrameResult


class Visualizer:
    """Draws combined emotion + posture overlays on frames."""

    def __init__(self, config: dict):
        self.show_bbox = config.get("show_bbox", True)
        self.show_comfort_bar = config.get("show_comfort_bar", True)
        self.show_emotion_text = config.get("show_emotion_text", True)
        self.show_gaze_text = config.get("show_gaze_text", True)
        self.show_posture_text = config.get("show_posture_text", True)
        self.show_depth_marker = config.get("show_depth_marker", True)
        self.show_state_warnings = config.get("show_state_warnings", True)
        self.show_fps = config.get("show_fps", True)
        self.bbox_thickness = config.get("bbox_thickness", 2)
        self._prev_time = time.time()
        self._fps = 0.0

    def _update_fps(self) -> None:
        now = time.time()
        dt = now - self._prev_time
        if dt > 0:
            self._fps = 0.3 * (1.0 / dt) + 0.7 * self._fps
        self._prev_time = now

    def _comfort_color(self, score: float) -> tuple[int, int, int]:
        """Green > 60, Yellow 30-60, Red < 30 (BGR)."""
        if score > 60:
            return (0, 200, 0)
        elif score > 30:
            return (0, 200, 200)
        else:
            return (0, 0, 200)

    def draw(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        self._update_fps()
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        score = result.integrated_comfort_score
        color = self._comfort_color(score)

        # --- Top: Integrated comfort bar ---
        if self.show_comfort_bar:
            bar_x, bar_y = 20, 30
            bar_w, bar_h = 200, 25
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * score / 100.0)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
            cv2.putText(overlay, f"Comfort: {score:.0f}/100", (bar_x, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Face bounding box ---
        if self.show_bbox and result.face_bbox is not None:
            bbox = result.face_bbox
            cv2.rectangle(overlay,
                          (int(bbox.x1), int(bbox.y1)),
                          (int(bbox.x2), int(bbox.y2)),
                          color, self.bbox_thickness)

        # --- Left column: Emotion + Gaze ---
        if self.show_emotion_text and result.emotion is not None:
            em = result.emotion
            cv2.putText(overlay,
                        f"{em.dominant_emotion} (V:{em.valence:+.2f} A:{em.arousal:+.2f})",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(overlay,
                        f"Emotion Score: {result.emotion_comfort_score:.0f}",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if self.show_gaze_text and result.gaze is not None:
            gz = result.gaze
            status = "Looking at camera" if gz.is_looking_at_camera else "Looking away"
            gaze_color = (0, 255, 0) if gz.is_looking_at_camera else (0, 0, 255)
            cv2.putText(overlay,
                        f"Gaze: {status} (Y:{gz.yaw:.1f} P:{gz.pitch:.1f})",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 2)

        # --- Right column: Posture ---
        if self.show_posture_text and result.pose is not None:
            pose = result.pose
            posture_color = (0, int(255 * pose.open_posture_score),
                             int(255 * (1 - pose.open_posture_score)))
            cv2.putText(overlay,
                        f"Posture: {pose.open_posture_score:.2f}",
                        (w - 260, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 2)

            if pose.interaction_z_meters is not None:
                cv2.putText(overlay,
                            f"Depth Z: {pose.interaction_z_meters:.2f}m",
                            (w - 260, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            cv2.putText(overlay,
                        f"Posture Score: {result.posture_comfort_score:.0f}",
                        (w - 260, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # --- Depth marker on closest wrist ---
        if self.show_depth_marker and result.pose is not None and result.pose.closest_wrist_px is not None:
            cv2.circle(overlay, result.pose.closest_wrist_px, 8, (255, 255, 0), -1)

        # --- Bottom center: State warnings ---
        if self.show_state_warnings and result.pose is not None:
            warning = None
            if result.pose.is_covering_mouth:
                warning = "STATE: SCARED (mouth/face covered)"
            elif result.pose.is_withdrawing:
                warning = "STATE: SUDDEN WITHDRAWAL"

            if warning:
                text_size = cv2.getTextSize(warning, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(overlay, warning, (text_x, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # --- FPS ---
        if self.show_fps:
            cv2.putText(overlay, f"FPS: {self._fps:.1f}", (w - 140, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return overlay
