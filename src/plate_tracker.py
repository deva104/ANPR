from collections import Counter, deque


class PlateTracker:
    """
    Manages multiple plate tracks across video frames.
    Between full-detection frames, boxes keep their last known positions
    (no OpenCV correlation tracker). Each track has:
      - A unique integer box_id
      - Current box in xyxy pixel coords
      - Rolling OCR text buffer (last N results)
      - Last accepted text (reused when OCR is gated)
      - Frame counter since last OCR run
      - Frame counter since last detection association
    """

    def __init__(self, vote_buffer=7, max_unseen_frames=30):
        self.vote_buffer = int(vote_buffer)
        self.max_unseen_frames = int(max_unseen_frames)
        self._next_box_id = 0
        self._tracks = {}

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)
        inter = inter_w * inter_h

        areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
        areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
        denom = float(areaA + areaB - inter)
        if denom <= 0.0:
            return 0.0
        return inter / denom

    def _new_track(self, frame, box):
        _ = frame  # kept for call-site compatibility
        box_id = self._next_box_id
        self._next_box_id += 1
        self._tracks[box_id] = {
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "ocr_buffer": deque(maxlen=self.vote_buffer),
            "last_text": "",
            "frames_since_ocr": 0,
            "frames_since_detect_match": 0,
        }

    def update_on_detect_frame(self, frame, detected_boxes):
        tracks_ids = list(self._tracks.keys())
        unmatched_tracks = set(tracks_ids)
        matched_tracks = set()

        for det in detected_boxes:
            best_id = None
            best_iou = 0.0
            for box_id in tracks_ids:
                if box_id in matched_tracks:
                    continue
                iou = self._iou(self._tracks[box_id]["box"], det)
                if iou >= 0.4 and iou > best_iou:
                    best_iou = iou
                    best_id = box_id

            if best_id is None:
                self._new_track(frame, det)
                continue

            t = self._tracks[best_id]
            t["box"] = [int(det[0]), int(det[1]), int(det[2]), int(det[3])]
            t["frames_since_detect_match"] = 0
            matched_tracks.add(best_id)
            unmatched_tracks.discard(best_id)

        to_delete = []
        for box_id in unmatched_tracks:
            self._tracks[box_id]["frames_since_detect_match"] += 1
            if self._tracks[box_id]["frames_since_detect_match"] > self.max_unseen_frames:
                to_delete.append(box_id)
        for box_id in to_delete:
            self._tracks.pop(box_id, None)

        return [(bid, self._tracks[bid]["box"]) for bid in sorted(self._tracks.keys())]

    def update_on_skip_frame(self, frame):
        _ = frame  # kept for API compatibility
        to_delete = []
        for box_id, tr in list(self._tracks.items()):
            tr["frames_since_detect_match"] += 1
            if tr["frames_since_detect_match"] > self.max_unseen_frames:
                to_delete.append(box_id)
        for box_id in to_delete:
            self._tracks.pop(box_id, None)

        return [(bid, self._tracks[bid]["box"]) for bid in sorted(self._tracks.keys())]

    def should_run_ocr(
        self,
        box_id,
        crop_w,
        crop_h,
        sharpness_score,
        ocr_min_width,
        ocr_min_height,
        ocr_sharpness_thresh,
        ocr_force_every,
    ):
        tr = self._tracks.get(box_id)
        if tr is None:
            return False, False
        quality_ok = (
            crop_w >= ocr_min_width
            and crop_h >= ocr_min_height
            and sharpness_score >= ocr_sharpness_thresh
        )
        forced = tr["frames_since_ocr"] >= ocr_force_every
        return bool(quality_ok or forced), bool(forced)

    def record_ocr_result(self, box_id, text):
        tr = self._tracks.get(box_id)
        if tr is None:
            return
        tr["ocr_buffer"].append(text)
        tr["frames_since_ocr"] = 0
        tr["last_text"] = text

    def get_voted_text(self, box_id):
        tr = self._tracks.get(box_id)
        if tr is None:
            return ""
        buf = list(tr["ocr_buffer"])
        if not buf:
            return ""
        if len(buf) >= 3:
            return Counter(buf).most_common(1)[0][0]
        return buf[-1]

    def get_last_text(self, box_id):
        tr = self._tracks.get(box_id)
        if tr is None:
            return ""
        return tr.get("last_text", "")

    def tick(self):
        for tr in self._tracks.values():
            tr["frames_since_ocr"] += 1
