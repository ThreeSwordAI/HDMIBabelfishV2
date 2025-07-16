import cv2

# If you have multiple video devices, try indices 0, 1, 2… or enumerate with ffmpeg:
# ffmpeg -list_devices true -f dshow -i dummy

DEVICE_INDEX = 0  # adjust if needed

# Open the capture stick via DirectShow
cap = cv2.VideoCapture(DEVICE_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"❌ Cannot open capture device at index {DEVICE_INDEX}")
    exit(1)

# (Optional) force 720p @30fps to match your PS3
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("▶️  Streaming PS3 feed — press ESC to quit")
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️  No frame received. Check PS3 output & cables.")
        break

    cv2.imshow("PS3 @ 720p Capture Test", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()

