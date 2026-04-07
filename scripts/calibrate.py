#!/usr/bin/env python3
"""Run integrated pipeline on all .bag files and report per-scenario comfort statistics.

Used to calibrate comfort scoring parameters (emotion, posture, and integrated weights).
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import IntegratedPipeline
from src.bag_source import BagSource


def find_bag_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.bag"))


def load_sidecar(bag_path: Path) -> dict | None:
    json_path = bag_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return None


def get_scenario(bag_path: Path) -> str:
    """Extract scenario folder name like sc01_walkby."""
    parts = bag_path.relative_to(PROJECT_ROOT / "data").parts
    return parts[0] if parts else "unknown"


def process_bag(bag_path: Path, pipeline: IntegratedPipeline) -> dict:
    """Process a bag file and return frame-level stats."""
    source = BagSource(str(bag_path), real_time=False)
    if not source.open():
        return {}

    pipeline.reset_state()

    emotion_scores = []
    posture_scores = []
    integrated_scores = []
    valences = []
    arousals = []
    face_detected_count = 0
    gaze_at_camera_count = 0
    pose_detected_count = 0
    mouth_cover_count = 0
    withdrawal_count = 0
    frame_count = 0

    try:
        while True:
            ret, frame, depth_frame, timestamp_ms = source.read()
            if not ret or frame is None:
                break

            result = pipeline.process_frame(frame, timestamp_ms, depth_frame=depth_frame)
            frame_count += 1

            emotion_scores.append(result.emotion_comfort_score)
            posture_scores.append(result.posture_comfort_score)
            integrated_scores.append(result.integrated_comfort_score)

            if result.face_detected:
                face_detected_count += 1
            if result.emotion:
                valences.append(result.emotion.valence)
                arousals.append(result.emotion.arousal)
            if result.gaze and result.gaze.is_looking_at_camera:
                gaze_at_camera_count += 1
            if result.pose and result.pose.has_pose:
                pose_detected_count += 1
                if result.pose.is_covering_mouth:
                    mouth_cover_count += 1
                if result.pose.is_withdrawing:
                    withdrawal_count += 1

            if frame_count % 50 == 0:
                print(f"    frame {frame_count}, "
                      f"integrated={result.integrated_comfort_score:.1f} "
                      f"(E:{result.emotion_comfort_score:.1f} P:{result.posture_comfort_score:.1f})",
                      flush=True)
    finally:
        source.release()

    if frame_count == 0:
        return {}

    # Comfort during interaction window (middle 50% of recording — where handoff happens)
    q1 = int(frame_count * 0.25)
    q3 = int(frame_count * 0.75)
    mid_emotion = emotion_scores[q1:q3] if q1 < q3 else emotion_scores
    mid_posture = posture_scores[q1:q3] if q1 < q3 else posture_scores
    mid_integrated = integrated_scores[q1:q3] if q1 < q3 else integrated_scores

    return {
        "frames": frame_count,
        "face_rate": face_detected_count / frame_count,
        "gaze_rate": gaze_at_camera_count / frame_count,
        "pose_rate": pose_detected_count / frame_count,
        "mouth_cover_rate": mouth_cover_count / frame_count,
        "withdrawal_rate": withdrawal_count / frame_count,
        "emotion_mean": float(np.mean(emotion_scores)),
        "emotion_mid": float(np.mean(mid_emotion)),
        "posture_mean": float(np.mean(posture_scores)),
        "posture_mid": float(np.mean(mid_posture)),
        "integrated_mean": float(np.mean(integrated_scores)),
        "integrated_mid": float(np.mean(mid_integrated)),
        "integrated_peak": float(np.max(integrated_scores)),
        "integrated_min": float(np.min(integrated_scores)),
        "valence_mean": float(np.mean(valences)) if valences else 0.0,
        "arousal_mean": float(np.mean(arousals)) if arousals else 0.0,
    }


def main():
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    bags = find_bag_files(PROJECT_ROOT / "data")
    if not bags:
        print("No .bag files found in data/")
        sys.exit(1)

    print(f"Found {len(bags)} .bag files. Loading models...\n")

    pipeline = IntegratedPipeline(config)
    pipeline.load_models()

    # Collect results by scenario
    scenario_results = {}

    for i, bag_path in enumerate(bags):
        scenario = get_scenario(bag_path)
        lighting = bag_path.parent.name
        name = f"{scenario}/{lighting}/{bag_path.name}"
        print(f"\n[{i+1}/{len(bags)}] {name}...", flush=True)

        stats = process_bag(bag_path, pipeline)
        if not stats:
            print("  SKIPPED")
            continue

        print(f"  => integrated_mid={stats['integrated_mid']:.1f} "
              f"(E:{stats['emotion_mid']:.1f} P:{stats['posture_mid']:.1f}), "
              f"peak={stats['integrated_peak']:.1f}, "
              f"V={stats['valence_mean']:+.2f}, A={stats['arousal_mean']:+.2f}, "
              f"face={stats['face_rate']:.0%}, pose={stats['pose_rate']:.0%}, "
              f"mouth={stats['mouth_cover_rate']:.0%}, withdraw={stats['withdrawal_rate']:.0%}",
              flush=True)

        if scenario not in scenario_results:
            scenario_results[scenario] = []
        scenario_results[scenario].append({"file": name, **stats})

    # Print summary
    print(f"\n{'='*120}")
    print("SCENARIO SUMMARY")
    print(f"{'='*120}")
    print(f"{'Scenario':<30} {'Files':>5} {'Integ(mid)':>11} {'Emot(mid)':>10} {'Post(mid)':>10} "
          f"{'Peak':>6} {'Face%':>6} {'Pose%':>6} {'Mouth%':>7} {'Wdraw%':>7}")
    print("-" * 120)

    for scenario in sorted(scenario_results.keys()):
        results = scenario_results[scenario]
        n = len(results)
        i_mid = np.mean([r["integrated_mid"] for r in results])
        e_mid = np.mean([r["emotion_mid"] for r in results])
        p_mid = np.mean([r["posture_mid"] for r in results])
        i_peak = np.mean([r["integrated_peak"] for r in results])
        face = np.mean([r["face_rate"] for r in results])
        pose = np.mean([r["pose_rate"] for r in results])
        mouth = np.mean([r["mouth_cover_rate"] for r in results])
        withdraw = np.mean([r["withdrawal_rate"] for r in results])

        print(f"{scenario:<30} {n:>5} {i_mid:>11.1f} {e_mid:>10.1f} {p_mid:>10.1f} "
              f"{i_peak:>6.1f} {face:>5.0%} {pose:>5.0%} {mouth:>6.0%} {withdraw:>6.0%}")

    print(f"\nConfig: emotion_weight={config['comfort']['emotion_weight']}, "
          f"posture_weight={config['comfort']['posture_weight']}, "
          f"gamma={config['comfort']['gamma']}, delta={config['comfort']['delta']}, "
          f"gaze_weight={config['comfort']['gaze_weight']}, "
          f"mouth_penalty={config['comfort']['mouth_cover_penalty']}, "
          f"withdrawal_penalty={config['comfort']['withdrawal_penalty']}, "
          f"ema_lambda={config['comfort']['ema_lambda']}")


if __name__ == "__main__":
    main()
