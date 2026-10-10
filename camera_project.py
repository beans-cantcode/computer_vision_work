import numpy as np
import cv2

print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    raise SystemExit

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))  # Larger kernel size

prev_frame = None
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray_frame
        continue

    abs_diff = cv2.absdiff(prev_frame, gray_frame)
    thresholds = cv2.threshold(abs_diff, 40, 255, cv2.THRESH_BINARY)[1]
    thresholds = cv2.dilate(thresholds, kernel, iterations=2)

    if cv2.countNonZero(thresholds) > 5000:
        count += 1
        print(f"Motion detected! {count}")

    contours, _ = cv2.findContours(thresholds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 10000:
            # Get the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the detected motion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Motion Detected", (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    prev_frame = gray_frame

    # Display the frame
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
