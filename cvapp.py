import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils


finger_tips = [4, 8, 12, 16, 20]  


cap = cv2.VideoCapture(0)


def detect_gesture(hand_type, lm_list, palm_facing_camera):
    fingers = []

    
    if hand_type == "Right":
        fingers.append(lm_list[4][0] > lm_list[3][0])  
    else:
        fingers.append(lm_list[4][0] < lm_list[3][0])  

    
    for tip_id in [8, 12, 16, 20]:
        fingers.append(lm_list[tip_id][1] < lm_list[tip_id - 2][1])

    total_fingers = fingers.count(True)

   
    if fingers == [False, True, True, False, False]:
        return "✌️ Peace (Palm In)" if palm_facing_camera else "✌️ Peace (Back Hand)"
    elif total_fingers == 0:
        return "👊 Fist"
    elif total_fingers == 5:
        return "🖐️ Open Hand (Palm In)" if palm_facing_camera else "🖐️ Open Hand (Back Hand)"
    elif fingers == [True, False, False, False, False]:
        return "👍 Thumbs Up"
    elif fingers == [False, True, False, False, False]:
        return "👆 Pointing Up"
    else:
        return f"{total_fingers} Fingers"


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            
            mp_hand_type = results.multi_handedness[idx].classification[0].label
            hand_type = "Right" if mp_hand_type == "Left" else "Left"

            
            lm_list = []
            for lm in hand_landmarks.landmark:
                h, w, _ = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            if lm_list:
                
                wrist_y = lm_list[0][1]
                middle_base_y = lm_list[9][1]
                palm_facing_camera = middle_base_y < wrist_y

                
                gesture = detect_gesture(hand_type, lm_list, palm_facing_camera)

                
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(img, f'{hand_type}: {gesture}', (10, 50 + idx * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                print(f"{hand_type} hand: {gesture}")

    cv2.imshow("Gesture & Palm Detector", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
