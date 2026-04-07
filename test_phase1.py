import os
import glob
import cv2
import torch
import numpy as np

from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
from src.object_detection.utils.utils import preprocess_image
from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference
from src.infer_video_utils import run_video_phase1

BENCHMARK_DETECT_EVERY = 5


def _fmt_min_sec(seconds):
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = seconds - 60 * m
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.1f}s"


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    demo_dir = os.path.join(root, "demo_videos")
    videos = sorted(glob.glob(os.path.join(demo_dir, "*.mp4")))
    if not videos:
        raise RuntimeError("No .mp4 found in demo_videos/")
    video_path = videos[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 0:
        total_frames = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps is None or fps <= 1e-3:
        fps = 25.0
    duration_seconds = total_frames / fps if fps > 0 else 0.0

    if duration_seconds <= 10:
        frame_cap = total_frames
    elif duration_seconds <= 60:
        frame_cap = int(fps * 30)
    else:
        frame_cap = int(fps * 20)

    frame_cap = max(frame_cap, 60)
    frame_cap = min(frame_cap, total_frames)

    print(f"Video: {video_path}")
    print(
        f"Total frames: {total_frames} | Duration: {duration_seconds:.1f}s | FPS: {fps:.1f}"
    )
    if frame_cap > 0:
        bench_vid_sec = frame_cap / fps
        print(
            f"Benchmark will process: {frame_cap} frames ({bench_vid_sec:.1f} seconds of video)"
        )
    else:
        print("Benchmark will process: 0 frames (empty or unreadable metadata)")
    print("Starting benchmark...")

    weights = os.path.join(root, "weights")
    od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
    od_model.load_state_dict(
        torch.load(
            os.path.join(weights, "best_od.pth"),
            map_location=torch.device("cpu"),
        )
    )
    lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
    lprnet.load_state_dict(
        torch.load(
            os.path.join(weights, "best_lprnet.pth"),
            map_location=torch.device("cpu"),
        )
    )
    if torch.cuda.is_available():
        od_model = od_model.cuda()
        lprnet = lprnet.cuda()

    def detector_fn(frame_bgr):
        image = preprocess_image(frame_bgr)
        if torch.cuda.is_available():
            image = image.cuda()
        with torch.no_grad():
            scores, classes, boxes = od_model(image)
        b = boxes[0].detach().cpu().numpy().tolist()
        s = scores[0].detach().cpu().numpy().tolist()
        out = []
        for idx, box in enumerate(b):
            score = float(s[idx]) if idx < len(s) else 1.0
            out.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), score])
        return out

    def ocr_fn(crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return ""
        im = cv2.resize(crop_bgr, (94, 24)).astype("float32")
        im -= 127.5
        im *= 0.0078125
        im = torch.from_numpy(np.transpose(im, (2, 0, 1)))
        labels = Greedy_Decode_inference(lprnet, torch.stack([im], 0))
        if not labels:
            return ""
        return labels[0]

    out_dir = os.path.join(root, "out_videos", "phase1_benchmark")
    stats = run_video_phase1(
        video_path=video_path,
        output_path=out_dir,
        detector_fn=detector_fn,
        ocr_fn=ocr_fn,
        detect_every=BENCHMARK_DETECT_EVERY,
        max_frames=frame_cap if frame_cap > 0 else None,
        time_breakdown=True,
    )

    processed = stats["total_frames"]
    bench_wall_sec = stats["total_time"]
    bench_vid_sec = processed / fps if fps > 0 else 0.0
    est_full_sec = (
        total_frames / max(1e-6, stats["avg_fps"]) if total_frames > 0 else 0.0
    )

    print("=== Phase 1 Benchmark Results ===")
    print(f"Video: {stats['video']}")
    print(
        f"Video duration: {duration_seconds:.1f}s | Total frames: {total_frames} | Native FPS: {fps:.1f}"
    )
    print(
        f"Benchmark processed: {processed} frames ({bench_vid_sec:.1f}s of video)"
    )
    print(
        f"Detection frames: {stats['detection_frames']} (ran detector every {BENCHMARK_DETECT_EVERY}th frame)"
    )
    print(f"OCR runs: {stats['ocr_runs']}")
    print(f"OCR skip rate: {stats['ocr_skip_rate']:.1f}%")
    print(f"Total wall time: {bench_wall_sec:.2f}s")
    print(f"Average FPS: {stats['avg_fps']:.1f}")
    print(f"Peak FPS: {stats['peak_fps']:.1f} (best 10-second window)")
    print(
        f"Estimated full video time at this rate: {_fmt_min_sec(est_full_sec)}"
    )
    print("================================")


if __name__ == "__main__":
    main()
