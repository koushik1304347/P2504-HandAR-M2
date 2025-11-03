import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe as mp
import math
import pywavefront
import datetime
import os
import numpy as np

# Directory to save screenshots
os.makedirs("screenshots", exist_ok=True)
screenshot_taken = False

# -------------------- HAND TRACKING SETUP --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

def get_pinch_strength(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    min_dist, max_dist = 0.02, 0.2
    return 1 - min(1, max(0, (distance - min_dist) / (max_dist - min_dist)))

def is_palm_open(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcps = [5, 9, 13, 17]
    extended = sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y for tip, mcp in zip(finger_tips, finger_mcps))
    return extended >= 3

def is_peace_sign(hand_landmarks):
    lm = hand_landmarks.landmark
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y
    return index_up and middle_up and ring_down and pinky_down

# NEW FUNCTION: Detect thumbs up gesture
def is_thumbs_up(hand_landmarks):
    lm = hand_landmarks.landmark
    # Thumb tip above thumb MCP (upward direction)
    thumb_up = lm[4].y < lm[3].y < lm[2].y
    # Other fingers folded
    fingers_folded = all(lm[tip].y > lm[base].y for tip, base in zip([8, 12, 16, 20], [5, 9, 13, 17]))
    return thumb_up and fingers_folded

# -------------------- PYGAME + OPENGL INIT --------------------
cap = cv2.VideoCapture(0)
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("Hand Controlled 3D Model (with Color)")
gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# -------------------- LOAD 3D MODEL --------------------
model_path = "model.obj"
scene = pywavefront.Wavefront(model_path, collect_faces=True, create_materials=True)

def draw_model(paused):
    glPushMatrix()
    glScalef(0.1, 0.1, 0.1)
    for mesh in scene.mesh_list:
        material = mesh.materials
        if hasattr(material, "diffuse"):
            color = material.diffuse
        else:
            color = [0.7, 0.7, 0.7]
        if paused:
            color = [c * 0.5 for c in color]
        glColor4f(*color, 1.0)
        glBegin(GL_TRIANGLES)
        for face in mesh.faces:
            for vertex_i in face:
                vertex = scene.vertices[vertex_i]
                glVertex3f(*vertex)
        glEnd()
    glPopMatrix()

# -------------------- SCREENSHOT FUNCTION --------------------
def take_screenshot():
    width, height = display
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = np.flipud(image)  # Flip vertically
    surface = pygame.image.frombuffer(image.tobytes(), (width, height), "RGB")

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join("screenshots", f"screenshot_{now}.png")
    pygame.image.save(surface, screenshot_path)
    print(f"📸 Screenshot saved: {screenshot_path}")

# -------------------- CONTROL VARIABLES --------------------
zoom_z = -5
translate_x = 0
translate_y = 0
rotate_x = 0
rotate_y = 0
paused = False

last_right_x = last_right_y = None
last_left_x = last_left_y = None
last_pinch_strength = 0
zoom_speed = 0.1

clock = pygame.time.Clock()
running = True

# -------------------- MAIN LOOP --------------------
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    right_hand, left_hand = None, None
    if result.multi_handedness:
        for idx, handedness in enumerate(result.multi_handedness):
            label = handedness.classification[0].label
            hand_landmarks = result.multi_hand_landmarks[idx]
            if label == "Right":
                right_hand = hand_landmarks
            elif label == "Left":
                left_hand = hand_landmarks

    # Pause if left hand is open
    paused = left_hand and is_palm_open(left_hand)

    # NEW: Realign model if right-hand thumbs up
    if right_hand and is_thumbs_up(right_hand):
        translate_x = 0
        translate_y = 0
        rotate_x = 0
        rotate_y = 0
        zoom_z = -5
        print("🔄 Model realigned to center!")

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glTranslatef(translate_x, translate_y, zoom_z)
    glRotatef(rotate_x, 1, 0, 0)
    glRotatef(rotate_y, 0, 1, 0)
    draw_model(paused)
    pygame.display.flip()

    # Take screenshot with right-hand peace sign when paused
    if paused and right_hand and is_peace_sign(right_hand) and not screenshot_taken:
        take_screenshot()
        screenshot_taken = True
    elif not paused:
        screenshot_taken = False

    # Controls active only when not paused
    if not paused:
        # Right hand controls rotation and zoom
        if right_hand:
            frame_h, frame_w, _ = frame.shape
            index_tip = right_hand.landmark[8]
            cx, cy = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
            if last_right_x is not None and last_right_y is not None:
                dx, dy = cx - last_right_x, cy - last_right_y
                rotate_y += dx * 0.5
                rotate_x += dy * 0.5
            last_right_x, last_right_y = cx, cy
            pinch_strength = get_pinch_strength(right_hand)
            zoom_z -= (pinch_strength - last_pinch_strength) * zoom_speed * 120
            zoom_z = max(-15, min(-0.5, zoom_z))
            last_pinch_strength = pinch_strength
        else:
            last_right_x = last_right_y = None
            last_pinch_strength = 0

        # Left hand controls translation
        if left_hand:
            frame_h, frame_w, _ = frame.shape
            index_tip = left_hand.landmark[8]
            cx, cy = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
            if last_left_x is not None and last_left_y is not None:
                dx = (cx - last_left_x) / frame_w
                dy = (cy - last_left_y) / frame_h
                translate_x += dx * 25
                translate_y -= dy * 25
            last_left_x, last_left_y = cx, cy
        else:
            last_left_x = last_left_y = None
    else:
        last_left_x = last_left_y = None
        last_right_x = last_right_y = None

    # -------------------- FEEDBACK TEXT --------------------
    if paused:
        if right_hand and is_peace_sign(right_hand):
            cv2.putText(frame, "📸 Screenshot Taken!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        else:
            cv2.putText(frame, "Paused (Show PEACE to capture)", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 4)

    if right_hand and is_thumbs_up(right_hand) and paused:
        cv2.putText(frame, "🔄 Model Realigned!", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 128, 0), 4)

    # -------------------- SHOW WEBCAM --------------------
    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
