import cv2
cap = cv2.VideoCapture("/home/babel-fish/Desktop/HDMIBabelfishV2/data/test_video/JapanRPG_TestSequence.mov")                 # or use VIDEO_PATH
if not cap.isOpened():
    raise RuntimeError("Cannot open camera/video")
cv2.namedWindow("imtest", cv2.WINDOW_AUTOSIZE)
while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow("imtest", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
