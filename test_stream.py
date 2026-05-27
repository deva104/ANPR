import cv2
import time

STREAM_URL = "http://10.186.230.167:81/stream"

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
        cap = cv2.VideoCapture(candidate)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            print(f"Stream connected: {candidate}")
            return cap
        cap.release()
    print("Failed to connect, retrying...")
    return None

cap = open_stream(STREAM_URL)
if cap is None:
    print("Cannot open stream at all - check IP address")
    exit()

prev = time.time()
frame_count = 0
reconnect_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print(f"Stream dropped. Reconnecting... (attempt {reconnect_count + 1})")
        cap.release()
        time.sleep(1)
        cap = open_stream(STREAM_URL)
        reconnect_count += 1
        if cap is None:
            time.sleep(2)
        continue

    now = time.time()
    fps = 1.0 / max(0.001, now - prev)
    prev = now
    frame_count += 1

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Frames: {frame_count} | Reconnects: {reconnect_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ESP32 Stream Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total frames: {frame_count} | Total reconnects: {reconnect_count}")
