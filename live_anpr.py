import cv2
import time
import threading
import queue
import torch
import sys
import os
import numpy as np

# Add project paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.object_detection.model.fcos import FCOSDetector
from src.object_detection.model.config import DefaultConfig
from src.object_detection.utils.utils import preprocess_image
from src.License_Plate_Recognition.model.LPRNet import build_lprnet
from src.License_Plate_Recognition.test_LPRNet import Greedy_Decode_inference
from src.plate_tracker import PlateTracker

STREAM_URL = "http://10.142.150.167:81/stream"
DETECT_EVERY = 1
DETECT_SCALE = 0.50
CONF_THRESH = 0.05
OCR_MIN_WIDTH = 80
OCR_MIN_HEIGHT = 20
OCR_SHARPNESS = 30.0
OCR_FORCE_EVERY = 20

# Latest frame buffer - size 1 means always get newest frame
frame_buffer = queue.Queue(maxsize=1)


def stream_reader():
    """Continuously reads stream, keeps only latest frame."""
    while True:
        cap = cv2.VideoCapture(STREAM_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            print("Reconnecting...")
            time.sleep(1)
            continue
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Stream dropped, reconnecting...")
                cap.release()
                break
            # Discard old frame if buffer full, put new one
            if frame_buffer.full():
                try:
                    frame_buffer.get_nowait()
                except queue.Empty:
                    pass
            frame_buffer.put(frame.copy())
        cap.release()
        time.sleep(0.5)


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

frame_idx = 0
prev_time = time.time()
inv_scale = 1.0 / max(1e-6, float(DETECT_SCALE))
detect_every = max(1, int(DETECT_EVERY))

while True:
    try:
        frame = frame_buffer.get(timeout=3)
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
        if voted_text and len(voted_text) >= 4:
            print(f"PLATE DETECTED: {voted_text} | frame: {frame_idx}")
        display_text = voted_text if voted_text else tracker.get_last_text(box_id)

        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            str(display_text),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
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
