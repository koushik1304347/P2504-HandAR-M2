import cv2
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_drawing = mp.solutions.drawing_utils

vertices = [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1]
]

edges = [
    (0,1), (1,2), (2,3), (3,0),
    (4,5), (5,6), (6,7), (7,4),
    (0,4), (1,5), (2,6), (3,7)
]

faces = [
    (0,1,2,3),
    (4,5,6,7),
    (0,1,5,4),
    (2,3,7,6),
    (1,2,6,5),
    (4,7,3,0)
]

colors = [
    (1.0, 0.0, 0.0, 0.5),
    (0.0, 1.0, 0.0, 0.5),
    (0.0, 0.0, 1.0, 0.5),
    (1.0, 1.0, 0.0, 0.5),
    (1.0, 0.0, 1.0, 0.5),
    (0.0, 1.0, 1.0, 0.5)
]

def draw_cube():
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glBegin(GL_QUADS)
    for i, face in enumerate(faces):
        glColor4fv(colors[i])
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

    glLineWidth(3)
    glColor3f(0, 0, 0)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()

def get_pinch_strength(hand_landmarks):
    
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    min_dist, max_dist = 0.02, 0.2
    pinch_strength = 1 - min(1, max(0, (distance - min_dist) / (max_dist - min_dist)))
    return pinch_strength

cap = cv2.VideoCapture(0)
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
glEnable(GL_DEPTH_TEST)


zoom_z = -7
translate_x = 0
translate_y = 0
rotate_x = 0
rotate_y = 0

last_pinch_strength = 0
zoom_speed = 0.1

last_left_x = None
last_left_y = None
last_right_x = None
last_right_y = None

clock = pygame.time.Clock()
running = True

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

    right_hand = None
    left_hand = None
    if result.multi_handedness:
        for idx, hand_handedness in enumerate(result.multi_handedness):
            label = hand_handedness.classification[0].label
            hand_landmarks = result.multi_hand_landmarks[idx]
            if label == "Right":
                right_hand = hand_landmarks
            elif label == "Left":
                left_hand = hand_landmarks

   
    if right_hand:
        frame_h, frame_w, _ = frame.shape
        index_tip = right_hand.landmark[8]
        cx, cy = int(index_tip.x * frame_w), int(index_tip.y * frame_h)

        if last_right_x is not None and last_right_y is not None:
            dx = cx - last_right_x
            dy = cy - last_right_y
            rotate_y += dx * 0.5
            rotate_x += dy * 0.5

        last_right_x, last_right_y = cx, cy

        pinch_strength = get_pinch_strength(right_hand)
        pinch_change = pinch_strength - last_pinch_strength
        zoom_z -= pinch_change * zoom_speed * 120
        zoom_z = max(-15, min(-0.5, zoom_z))
        last_pinch_strength = pinch_strength
    else:
        last_pinch_strength = 0
        last_right_x = last_right_y = None

   
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

    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
    glTranslatef(translate_x, translate_y, zoom_z)
    glRotatef(rotate_x, 1, 0, 0)
    glRotatef(rotate_y, 0, 1, 0)
    draw_cube()

    pygame.display.flip()
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()
