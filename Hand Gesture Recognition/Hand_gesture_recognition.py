import cv2
import os

# List of gestures
gestures = ["Open_Hand", "Fist", "Peace", "Thumbs_Up"]
total_images = 1 

# Create folders
for gesture in gestures:
    os.makedirs(f"hand_dataset/{gesture}", exist_ok=True)

cap = cv2.VideoCapture(0)

for gesture in gestures:
    print(f"\nStarting capture for gesture: {gesture}")
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        x, y, w, h = 100, 100, 300, 300  # ROI
        roi = frame[y:y+h, x:x+w]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        cv2.putText(frame, f"Images Captured: {count}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        cv2.imshow("Hand Gesture Dataset", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            img_name = f"hand_dataset/{gesture}/{count}.jpg"
            cv2.imwrite(img_name, roi)
            count += 1
            print(f"Saved {img_name}")

        if key == ord('q') or count >= total_images:
            break

cap.release()
cv2.destroyAllWindows()
print("Dataset capture complete!")