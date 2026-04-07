from .types import FrameResult


class IntegratedComfortScorer:
    """Computes comfort from both emotion and posture signals.

    Produces three scores:
    - Emotion comfort: valence/arousal/gaze based (0-100)
    - Posture comfort: open posture with penalties for mouth cover / withdrawal (0-100)
    - Integrated: weighted blend of both
    """

    def __init__(self, config: dict):
        # Emotion weights
        self.gamma = config.get("gamma", 0.5)
        self.delta = config.get("delta", 0.3)
        self.gaze_weight = config.get("gaze_weight", 0.6)

        # Posture weights
        self.mouth_cover_penalty = config.get("mouth_cover_penalty", 30.0)
        self.withdrawal_penalty = config.get("withdrawal_penalty", 25.0)

        # Integration weights
        self.emotion_weight = config.get("emotion_weight", 0.55)
        self.posture_weight = config.get("posture_weight", 0.45)

        # EMA smoothing
        self.ema_lambda = config.get("ema_lambda", 0.85)
        self.no_face_decay_rate = config.get("no_face_decay_rate", 0.005)
        self.no_face_decay_target = config.get("no_face_decay_target", 35.0)
        self.no_pose_decay_rate = config.get("no_pose_decay_rate", 0.01)
        self.no_pose_decay_target = config.get("no_pose_decay_target", 50.0)

        self._smoothed_emotion = 50.0
        self._smoothed_posture = 50.0

    def _compute_emotion_score(self, result: FrameResult) -> float:
        """Compute instant emotion comfort score (0-100)."""
        if not result.face_detected or result.emotion is None:
            self._smoothed_emotion += (
                (self.no_face_decay_target - self._smoothed_emotion) * self.no_face_decay_rate
            )
            return self._smoothed_emotion

        gaze_signal = 0.0
        if result.gaze is not None and result.gaze.is_looking_at_camera:
            gaze_signal = 1.0

        raw = self.gamma * result.emotion.valence - self.delta * result.emotion.arousal + self.gaze_weight * gaze_signal
        max_mag = self.gamma + self.delta + self.gaze_weight
        instant = ((raw + max_mag) / (2 * max_mag)) * 100.0
        instant = max(0.0, min(100.0, instant))

        self._smoothed_emotion = self.ema_lambda * instant + (1 - self.ema_lambda) * self._smoothed_emotion
        return self._smoothed_emotion

    def _compute_posture_score(self, result: FrameResult) -> float:
        """Compute instant posture comfort score (0-100)."""
        if result.pose is None or not result.pose.has_pose:
            self._smoothed_posture += (
                (self.no_pose_decay_target - self._smoothed_posture) * self.no_pose_decay_rate
            )
            return self._smoothed_posture

        instant = result.pose.open_posture_score * 100.0

        if result.pose.is_covering_mouth:
            instant -= self.mouth_cover_penalty
        if result.pose.is_withdrawing:
            instant -= self.withdrawal_penalty

        instant = max(0.0, min(100.0, instant))

        self._smoothed_posture = self.ema_lambda * instant + (1 - self.ema_lambda) * self._smoothed_posture
        return self._smoothed_posture

    def update(self, result: FrameResult) -> tuple[float, float, float]:
        """Update and return (emotion_score, posture_score, integrated_score)."""
        emotion = self._compute_emotion_score(result)
        posture = self._compute_posture_score(result)
        integrated = self.emotion_weight * emotion + self.posture_weight * posture
        return emotion, posture, integrated

    def reset(self) -> None:
        self._smoothed_emotion = 50.0
        self._smoothed_posture = 50.0
