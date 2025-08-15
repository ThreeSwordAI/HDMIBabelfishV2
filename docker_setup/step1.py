import cv2

# Video path (adjust as necessary based on your directory)
VIDEO_PATH = "/workspace/data/test_video/JapanRPG_TestSequence.mov"

# Open the video file
cap = cv2.VideoCapture(VIDEO_PATH)

# Check if video opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open video")
    exit()

# Display the video frames
while True:
    ret, frame = cap.read()

    # If the frame was read correctly
    if not ret:
        print("📺 End of video reached")
        break

    # Show the frame in a window
    cv2.imshow("Video Display", frame)

    # Wait for key press; exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👤 User quit")
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
