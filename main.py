from ursina import *
import cv2
import mediapipe as mp
import math
import numpy as np
import datetime
import os


# ============================================
# 1) Ursina Setup
# ============================================
app = Ursina()
window.color = color.black

# Load GLB model
car = Entity(
    model='vintage_racing_car.glb',   # your GLB
    scale=1,
    origin=Vec3(0, 0, 0),
    position=Vec3(0, 0, 0)
)

# Initial camera position
camera.position = Vec3(0, 0, -6)
camera.look_at(car.position)

# Screenshot folder
os.makedirs("screenshots", exist_ok=True)


# ============================================
# 2) Mediapipe Setup
# ============================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

cap = cv2.VideoCapture(0)


# ============================================
# 3) Gesture Helpers
# ============================================
def get_pinch_strength(hand_landmarks):
    t = hand_landmarks.landmark[4]
    i = hand_landmarks.landmark[8]
    d = math.dist((t.x, t.y), (i.x, i.y))
    return max(0, min(1, 1 - (d - 0.02) / 0.2))


def is_open_hand(hand):
    lm = hand.landmark
    tips = [8, 12, 16, 20]
    base = [5, 9, 13, 17]
    extended = sum(lm[t].y < lm[b].y for t, b in zip(tips, base))
    thumb_open = abs(lm[4].x - lm[3].x) > 0.05
    return extended >= 4 and thumb_open


def is_peace(hand):
    lm = hand.landmark
    return (lm[8].y < lm[6].y and
            lm[12].y < lm[10].y and
            lm[16].y > lm[14].y and
            lm[20].y > lm[18].y)


def is_thumbs_up(hand):
    lm = hand.landmark
    thumb_up = lm[4].y < lm[3].y < lm[2].y
    fingers_folded = all(lm[t].y > lm[b].y for t, b in zip([8,12,16,20],[5,9,13,17]))
    return thumb_up and fingers_folded


# ============================================
# 4) Control Variables
# ============================================
paused = False
last_right_x = last_right_y = None
last_left_x = last_left_y = None
last_pinch = 0
screenshot_done = False


# ============================================
# 5) Screenshot Function
# ============================================
def take_screenshot():
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"screenshots/snap_{now}.png"
    window.screenshot(name=path)
    print("📸 Saved:", path)


# ============================================
# 6) UPDATE LOOP
# ============================================
def update():
    global paused, last_right_x, last_right_y
    global last_left_x, last_left_y, last_pinch
    global screenshot_done

    # get cam frame
    ok, frame = cap.read()
    if not ok:
        return

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    # detect hands
    left = right = None
    if res.multi_hand_landmarks and res.multi_handedness:
        for i, h in enumerate(res.multi_handedness):
            label = h.classification[0].label
            lm = res.multi_hand_landmarks[i]
            if label == "Right":
                right = lm
            else:
                left = lm

    # Pause if left hand open
    paused = left and is_open_hand(left)

    # Reset (when paused)
    if paused and right and is_thumbs_up(right):
        car.position = Vec3(0, 0, 0)
        car.rotation = Vec3(0, 0, 0)
        camera.position = Vec3(0, 0, -6)
        print("🔄 Reset")

    # Screenshot
    if paused and right and is_peace(right) and not screenshot_done:
        take_screenshot()
        screenshot_done = True
    if not paused:
        screenshot_done = False

    # Controls only when not paused
    if not paused:
        h, w, _ = frame.shape

        # RIGHT HAND → rotate + zoom
        if right:
            rx = int(right.landmark[8].x * w)
            ry = int(right.landmark[8].y * h)

            if last_right_x is not None:
                dx = rx - last_right_x
                dy = ry - last_right_y
                car.rotation_y += dx * 0.4
                car.rotation_x -= dy * 0.4

            last_right_x, last_right_y = rx, ry

            pinch = get_pinch_strength(right)
            zoom = (last_pinch - pinch) * 2

            # Fix: old Ursina requires full Vec3 update
            camera.position = Vec3(
                camera.position.x,
                camera.position.y,
                camera.position.z + zoom
            )

            last_pinch = pinch
        else:
            last_right_x = last_right_y = None
            last_pinch = 0

        # LEFT HAND → translation
        if left:
            lx = int(left.landmark[8].x * w)
            ly = int(left.landmark[8].y * h)

            if last_left_x is not None:
                dx = (lx - last_left_x) / w
                dy = (ly - last_left_y) / h
                car.position = Vec3(
                    car.position.x + dx * 3,
                    car.position.y - dy * 3,
                    car.position.z
                )

            last_left_x, last_left_y = lx, ly
        else:
            last_left_x = last_left_y = None

    # show webcam
    cv2.imshow("Hand Tracking", frame)
    cv2.waitKey(1)


# ============================================
# 7) RUN APP
# ============================================
app.run()
cap.release()
cv2.destroyAllWindows()
