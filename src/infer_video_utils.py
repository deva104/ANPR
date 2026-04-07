"""Shared helpers for infer_objectdet.py and infer_semanticseg.py video I/O."""

import os
import json
import time
from time import perf_counter
import cv2
from src.plate_tracker import PlateTracker

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def is_video_file(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def safe_fps(capture, default=25.0):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3 or fps > 240.0:
        return default
    return float(fps)


def video_writer(output_path, fps, frame_size):
    """Open a VideoWriter; prefer MP4 (mp4v), fall back to MJPG AVI."""
    d = os.path.dirname(os.path.abspath(output_path))
    if d:
        os.makedirs(d, exist_ok=True)
    w, h = int(frame_size[0]), int(frame_size[1])
    size = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)
    if writer.isOpened():
        return writer, output_path
    writer.release()
    base, _ = os.path.splitext(output_path)
    fallback = base + ".avi"
    writer = cv2.VideoWriter(
        fallback, cv2.VideoWriter_fourcc(*"MJPG"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(
            "Could not open VideoWriter for MP4 or AVI. "
            "Install OpenCV with codec support or try a different output path."
        )
    return writer, fallback


def annotated_video_path(video_path, output_dir):
    """Path for annotated video under output_dir/videos/."""
    sub = os.path.join(output_dir, "videos")
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(sub, f"{stem}_annotated.mp4")


def run_video_phase1(
    video_path,
    output_path,
    detector_fn,  # callable: (frame_bgr) -> list of [x1,y1,x2,y2,score]
    ocr_fn,  # callable: (crop_bgr) -> str
    detect_every=5,
    detect_scale=0.67,
    ocr_min_width=100,
    ocr_min_height=28,
    ocr_sharpness=80.0,
    ocr_force_every=10,
    vote_buffer=7,
    live_preview=False,
    max_frames=None,
    conf_thresh=0.1,
    time_breakdown=False,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = safe_fps(cap, default=25.0)

    tracker = PlateTracker(vote_buffer=vote_buffer)

    videos_dir = os.path.join(output_path, "videos")
    json_dir = os.path.join(output_path, "jsons")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(videos_dir, f"{stem}_annotated.mp4")
    writer = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H),
    )
    if not writer.isOpened():
        writer.release()
        writer, out_video_path = video_writer(out_video_path, fps, (W, H))

    frame_idx = 0
    detect_count = 0
    ocr_count = 0
    json_results = []
    t_start = time.time()
    frame_times = []

    inv_scale = 1.0 / max(1e-6, float(detect_scale))
    detect_every = max(1, int(detect_every))

    t_read = t_detect = t_track = t_ocr = t_draw = t_write = 0.0
    t_loop_start = perf_counter() if time_breakdown else None

    while True:
        if time_breakdown:
            _t0 = perf_counter()
        ret, frame = cap.read()
        if time_breakdown:
            t_read += perf_counter() - _t0
        if not ret:
            break

        frame_start = time.time()
        tracker.tick()

        if frame_idx % detect_every == 0:
            small = cv2.resize(frame, (max(1, int(W * detect_scale)), max(1, int(H * detect_scale))))
            if time_breakdown:
                _t0 = perf_counter()
            raw_boxes = detector_fn(small) or []
            if time_breakdown:
                t_detect += perf_counter() - _t0

            scaled_boxes = []
            for it in raw_boxes:
                if len(it) >= 5:
                    x1, y1, x2, y2, score = it[:5]
                else:
                    x1, y1, x2, y2 = it[:4]
                    score = 1.0
                if score is not None and float(score) < float(conf_thresh):
                    continue
                X1 = int(max(0, min(W - 1, round(float(x1) * inv_scale))))
                Y1 = int(max(0, min(H - 1, round(float(y1) * inv_scale))))
                X2 = int(max(0, min(W - 1, round(float(x2) * inv_scale))))
                Y2 = int(max(0, min(H - 1, round(float(y2) * inv_scale))))
                if X2 <= X1 or Y2 <= Y1:
                    continue
                scaled_boxes.append([X1, Y1, X2, Y2])
            if time_breakdown:
                _t0 = perf_counter()
            active_tracks = tracker.update_on_detect_frame(frame, scaled_boxes)
            if time_breakdown:
                t_track += perf_counter() - _t0
            detect_count += 1
        else:
            if time_breakdown:
                _t0 = perf_counter()
            active_tracks = tracker.update_on_skip_frame(frame)
            if time_breakdown:
                t_track += perf_counter() - _t0

        annotated_frame = frame.copy()
        for box_id, box in active_tracks:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1 = max(0, min(W - 1, x1))
            y1 = max(0, min(H - 1, y1))
            x2 = max(0, min(W - 1, x2))
            y2 = max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            sharpness = cv2.Laplacian(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var()

            should_ocr, is_forced = tracker.should_run_ocr(
                box_id,
                crop.shape[1],
                crop.shape[0],
                sharpness,
                ocr_min_width,
                ocr_min_height,
                ocr_sharpness,
                ocr_force_every,
            )

            if should_ocr:
                if time_breakdown:
                    _t0 = perf_counter()
                raw_text = ocr_fn(crop)
                if time_breakdown:
                    t_ocr += perf_counter() - _t0
                tracker.record_ocr_result(box_id, raw_text)
                ocr_count += 1
            else:
                raw_text = None

            voted_text = tracker.get_voted_text(box_id)
            display_text = voted_text if voted_text else tracker.get_last_text(box_id)

            json_results.append(
                {
                    "frame": frame_idx,
                    "box_id": box_id,
                    "box": [x1, y1, x2, y2],
                    "raw_ocr": raw_text,
                    "voted_text": voted_text,
                    "ocr_forced": is_forced,
                }
            )

            if time_breakdown:
                _t0 = perf_counter()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                str(display_text),
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if time_breakdown:
                t_draw += perf_counter() - _t0

        if frame_idx % 50 == 0 and frame_idx > 0:
            elapsed = time.time() - t_start
            avg_fps = frame_idx / max(1e-6, elapsed)
            print(f"Frame {frame_idx} | Detects: {detect_count} | OCR runs: {ocr_count} | Avg FPS: {avg_fps:.1f}")

        if time_breakdown:
            _t0 = perf_counter()
        writer.write(annotated_frame)
        if time_breakdown:
            t_write += perf_counter() - _t0
        if live_preview:
            cv2.imshow("ANPR Phase1", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_times.append(time.time() - frame_start)
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()
    if live_preview:
        cv2.destroyAllWindows()

    out_json = os.path.join(json_dir, f"{stem}.json")
    with open(out_json, "w") as f:
        json.dump(json_results, f)

    total_time = time.time() - t_start
    track_events = len(json_results)
    ocr_skip_rate = 0.0
    if track_events > 0:
        ocr_skip_rate = 100.0 * max(0, track_events - ocr_count) / track_events
    avg_fps = frame_idx / max(1e-6, total_time)
    peak_fps = 0.0
    if frame_times:
        window = max(1, int(round(fps * 10)))
        window = min(window, len(frame_times))
        best = 0.0
        run = sum(frame_times[:window])
        if run > 0:
            best = max(best, window / run)
        for i in range(window, len(frame_times)):
            run += frame_times[i] - frame_times[i - window]
            if run > 0:
                best = max(best, window / run)
        peak_fps = best

    print(
        f"Phase1 done | total frames: {frame_idx} | detect runs: {detect_count} | "
        f"ocr runs: {ocr_count} | ocr skip rate: {ocr_skip_rate:.1f}% | "
        f"time: {total_time:.2f}s | avg FPS: {avg_fps:.2f}"
    )

    if time_breakdown:
        t_loop_wall = perf_counter() - t_loop_start

        def _pct(sec, denom):
            return 100.0 * sec / denom if denom > 1e-9 else 0.0

        sum_labeled = t_read + t_detect + t_track + t_ocr + t_draw + t_write
        t_other = max(0.0, t_loop_wall - sum_labeled)
        print()
        print("=== TIME BREAKDOWN ===")
        print(f"Total wall time (main loop): {t_loop_wall:.2f}s")
        print(f"  cap.read():        {t_read:8.2f}s  ({_pct(t_read, t_loop_wall):5.1f}%)")
        print(f"  detector:          {t_detect:8.2f}s  ({_pct(t_detect, t_loop_wall):5.1f}%)")
        print(f"  tracker update:    {t_track:8.2f}s  ({_pct(t_track, t_loop_wall):5.1f}%)")
        print(f"  ocr:               {t_ocr:8.2f}s  ({_pct(t_ocr, t_loop_wall):5.1f}%)")
        print(f"  draw:              {t_draw:8.2f}s  ({_pct(t_draw, t_loop_wall):5.1f}%)")
        print(f"  writer.write():    {t_write:8.2f}s  ({_pct(t_write, t_loop_wall):5.1f}%)")
        print(f"  other:             {t_other:8.2f}s  ({_pct(t_other, t_loop_wall):5.1f}%)")
        print(f"Detection frames ran: {detect_count} out of {frame_idx} total frames")
        print(f"OCR runs: {ocr_count}")
        print(f"Full run wall time (incl. setup/teardown): {total_time:.2f}s")
        print("======================")

    return {
        "video": os.path.basename(video_path),
        "total_frames": frame_idx,
        "detection_frames": detect_count,
        "ocr_runs": ocr_count,
        "ocr_skip_rate": ocr_skip_rate,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "peak_fps": peak_fps,
        "output_video": out_video_path,
        "output_json": out_json,
    }
