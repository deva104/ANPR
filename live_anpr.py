import argparse
import cv2
import time
import threading
import queue
import torch
import sys
import os
import numpy as np
import requests
from threading import Thread

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
from src.object_detection.utils.utils import preprocess_image
from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference
from src.plate_tracker import PlateTracker

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source",
    type=str,
    default="http://10.162.159.167:81/stream",
    help="ESP32-CAM stream URL",
)
parser.add_argument(
    "--mjpeg",
    action="store_true",
    help="Use MJPEG capture backend explicitly",
)
parser.add_argument(
    "--backend",
    type=str,
    default="http://localhost:8000",
    help="Backend URL",
)
args = parser.parse_args()
STREAM_URL = args.source
BACKEND_URL = args.backend.rstrip("/")
DETECT_EVERY = 1
DETECT_SCALE = 0.50
CONF_THRESH = 0.05
OCR_MIN_WIDTH = 80
OCR_MIN_HEIGHT = 20
OCR_SHARPNESS = 30.0
OCR_FORCE_EVERY = 20

# Latest frame buffer - size 1 means always get newest frame
frame_buffer = queue.Queue(maxsize=1)


def send_to_backend(plate_number, box_id):
    """Send verified plate to backend in a separate thread."""

    def _send():
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/verify",
                json={"plate_number": plate_number},
                timeout=5,
            )
            if response.status_code == 200:
                result = response.json()
                granted = result.get("granted", False)
                owner = result.get("owner_name", "Unknown")
                vtype = result.get("vehicle_type", "")
                reason = result.get("reason", "")
                t = time.strftime("%H:%M:%S")
                track_results[plate_number] = {
                    "granted": granted,
                    "label": f"{'ALLOWED' if granted else 'DENIED'}: {plate_number}",
                }
                if granted:
                    print(f"[ALLOWED] {plate_number} | {owner} | {vtype} | {t}")
                else:
                    print(f"[DENIED]  {plate_number} | {owner} | {reason} | {t}")
            else:
                print(f"[BACKEND ERROR] Status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"[BACKEND OFFLINE] Plate: {plate_number} (not sent)")
        except requests.exceptions.Timeout:
            print(f"[BACKEND TIMEOUT] Plate: {plate_number} (not sent)")
        except Exception as e:
            print(f"[BACKEND ERROR] {plate_number}: {e}")

    Thread(target=_send, daemon=True).start()


def stream_candidates(url):
    base = (url or "").strip()
    if not base:
        return []
    no_slash = base.rstrip("/")
    return [
        base,
        f"{no_slash}/stream",
        f"{no_slash}:81/stream",
        f"{no_slash}/video",
        f"{no_slash}/mjpeg",
    ]


def open_stream(url):
    for candidate in stream_candidates(url):
        if args.mjpeg:
            cap = cv2.VideoCapture(candidate, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(candidate)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 10)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        if cap.isOpened():
            # Try reading one test frame to confirm stream is live
            ret, _ = cap.read()
            if ret:
                print(f"Stream confirmed: {candidate}")
                return cap
        cap.release()
    return None


def stream_reader():
    consecutive_failures = 0
    while True:
        cap = open_stream(STREAM_URL)
        if cap is None or not cap.isOpened():
            consecutive_failures += 1
            wait = min(consecutive_failures * 1, 5)
            print(f"Reconnecting in {wait}s...")
            time.sleep(wait)
            continue
        consecutive_failures = 0
        fail_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                fail_count += 1
                if fail_count >= 5:
                    print("Stream dropped, reconnecting...")
                    break
                time.sleep(0.1)
                continue
            fail_count = 0
            if frame_buffer.full():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            frame_buffer.put(frame.copy())
        cap.release()
        time.sleep(1)


_ROOT = os.path.dirname(os.path.abspath(__file__))
_WEIGHTS = os.path.join(_ROOT, "weights")

od_model = FCOSDetector(mode="inference", config=DefaultConfig).eval()
od_model.load_state_dict(
    torch.load(
        os.path.join(_WEIGHTS, "best_od.pth"),
        map_location=torch.device("cpu"),
    )
)

lprnet = build_lprnet(lpr_max_len=16, class_num=37).eval()
lprnet.load_state_dict(
    torch.load(
        os.path.join(_WEIGHTS, "best_lprnet.pth"),
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
        out.append(
            [
                int(box[0]),
                int(box[1]),
                int(box[2]),
                int(box[3]),
                score,
            ]
        )
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


# Start stream reader thread
reader_thread = threading.Thread(target=stream_reader, daemon=True)
reader_thread.start()

print("Waiting for stream...")
time.sleep(2)

tracker = PlateTracker(vote_buffer=12, max_unseen_frames=8)

track_results = {}
# Stores {box_id: {"granted": bool, "label": str}}

frame_idx = 0
prev_time = time.time()
inv_scale = 1.0 / max(1e-6, float(DETECT_SCALE))
detect_every = max(1, int(DETECT_EVERY))

while True:
    try:
        frame = frame_buffer.get(timeout=10)
    except queue.Empty:
        print("No frame received")
        continue

    H, W = frame.shape[:2]

    # Sharpen the frame before detection
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    frame = cv2.filter2D(frame, -1, kernel)

    display = frame.copy()

    tracker.tick()

    if frame_idx % detect_every == 0:
        small = cv2.resize(
            frame,
            (max(1, int(W * DETECT_SCALE)), max(1, int(H * DETECT_SCALE))),
        )
        raw_boxes = detector_fn(small) or []

        scaled_boxes = []
        for it in raw_boxes:
            if len(it) >= 5:
                x1, y1, x2, y2, score = it[:5]
            else:
                x1, y1, x2, y2 = it[:4]
                score = 1.0
            if score is not None and float(score) < float(CONF_THRESH):
                continue
            X1 = int(max(0, min(W - 1, round(float(x1) * inv_scale))))
            Y1 = int(max(0, min(H - 1, round(float(y1) * inv_scale))))
            X2 = int(max(0, min(W - 1, round(float(x2) * inv_scale))))
            Y2 = int(max(0, min(H - 1, round(float(y2) * inv_scale))))
            if X2 <= X1 or Y2 <= Y1:
                continue
            scaled_boxes.append([X1, Y1, X2, Y2])
        active_tracks = tracker.update_on_detect_frame(frame, scaled_boxes)
    else:
        active_tracks = tracker.update_on_skip_frame(frame)

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

        should_ocr, _is_forced = tracker.should_run_ocr(
            box_id,
            crop.shape[1],
            crop.shape[0],
            sharpness,
            OCR_MIN_WIDTH,
            OCR_MIN_HEIGHT,
            OCR_SHARPNESS,
            OCR_FORCE_EVERY,
        )

        if should_ocr:
            raw_text = ocr_fn(crop)
            tracker.record_ocr_result(box_id, raw_text)

        voted_text = tracker.get_voted_text(box_id)
        display_text = voted_text if voted_text else tracker.get_last_text(box_id)

        ready = tracker.is_ready_for_verification(box_id)

        if ready:
            tracker.mark_verified(box_id, cooldown_seconds=5.0)
            send_to_backend(voted_text, box_id)
            box_color = (0, 255, 0)
            label = f"VERIFIED: {voted_text}"

        else:
            cooldown_left = tracker.get_cooldown_remaining(box_id)
            if cooldown_left > 0:
                cached = track_results.get(voted_text)
                if cached is not None:
                    if cached["granted"]:
                        box_color = (0, 255, 0)
                    else:
                        box_color = (0, 0, 255)
                    label = cached["label"]
                else:
                    box_color = (0, 255, 255)
                    label = f"{display_text} ({cooldown_left:.1f}s)"
            else:
                box_color = (255, 255, 255)
                label = display_text

        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(
            display,
            str(label),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_color,
            2,
            cv2.LINE_AA,
        )

    now = time.time()
    fps = 1.0 / max(0.001, now - prev_time)
    prev_time = now
    cv2.putText(
        display,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Live ANPR", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_idx += 1

cv2.destroyAllWindows()
