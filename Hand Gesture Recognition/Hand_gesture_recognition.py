import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Tip landmarks
tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

def fingers_up(hand_landmarks, hand_label):
    lm = hand_landmarks.landmark
    fingers = []

    # Thumb: Compare tip with IP joint (landmark 3)
    if hand_label == "Right":
        fingers.append(1 if lm[tip_ids[0]].x > lm[tip_ids[0]-1].x + 0.02 else 0)
    else:  # Left hand
        fingers.append(1 if lm[tip_ids[0]].x < lm[tip_ids[0]-1].x - 0.02 else 0)

    # Other fingers: tip vs PIP with small threshold
    for id in range(1,5):
        fingers.append(1 if lm[tip_ids[id]].y < lm[tip_ids[id]-2].y - 0.01 else 0)

    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    gesture_text = "No hand detected"

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            hand_label = hand_handedness.classification[0].label
            fingers = fingers_up(hand_lms, hand_label)
            total_fingers = fingers.count(1)

            # Robust gesture mapping
            if total_fingers == 0:
                gesture_text = "Fist (Closed hand)"
            elif total_fingers >= 4:  # Treat 4 or 5 fingers as Open hand
                gesture_text = "Open hand"
            else:
                gesture_text = f"{total_fingers} fingers"

    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()