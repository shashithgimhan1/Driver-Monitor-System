import os
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import platform

alarm_active = False

def sound_alarm():
    global alarm_active
    current_os = platform.system()
    
    while alarm_active:
        if current_os == "Windows":
            import winsound
            winsound.Beep(1500, 400)
        else:
            time.sleep(0.1)

def trigger_alarm(state):
    global alarm_active
    if state and not alarm_active:
        alarm_active = True
        t = threading.Thread(target=sound_alarm, daemon=True)
        t.start()
    elif not state and alarm_active:
        alarm_active = False

# MEDIAPIPE INITIALIZATION
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils          

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True, 
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Landmark Mappings
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402, 14, 178]
NOSE_TIP = 4
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

EAR_THRESHOLD = 0.15       
MAR_THRESHOLD = 0.85       
DROWSY_FRAME_LIMIT = 30    
YAWN_FRAME_LIMIT = 35      
DISTRACTION_FRAME_LIMIT = 50

eye_counter = 0
yawn_counter = 0
distraction_counter = 0

def calculate_ear(landmarks, eye_indices, w, h):
    pts = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in eye_indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h_dist = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h_dist)

def calculate_mar(landmarks, mouth_indices, w, h):
    pts = [np.array([landmarks[idx].x * w, landmarks[idx].y * h]) for idx in mouth_indices]
    v = np.linalg.norm(pts[2] - pts[6])
    h_dist = np.linalg.norm(pts[0] - pts[4])
    return v / h_dist

def check_head_pose(landmarks, w):
    nose_x = landmarks[NOSE_TIP].x * w
    left_cheek_x = landmarks[LEFT_CHEEK].x * w
    right_cheek_x = landmarks[RIGHT_CHEEK].x * w
    
    dist_left = abs(nose_x - left_cheek_x)
    dist_right = abs(right_cheek_x - nose_x)
    
    ratio = dist_left / (dist_right + 1e-6)

    if ratio < 0.5:   
        return "Looking Left", True
    elif ratio > 2.0:
        return "Looking Right", True
    return "Center", False


cap = cv2.VideoCapture(0)
print("Starting Camera... Press 'q' inside the video window to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    drowsy_alert = False
    yawn_alert = False
    distraction_alert = False
    gaze_status = "Center"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            custom_mesh_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=custom_mesh_style, # Colors the dots
                connection_drawing_spec=custom_mesh_style # Colors the lines connecting the dots
            )

            # Calculate metrics
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(face_landmarks.landmark, MOUTH, w, h)
            gaze_status, is_distracted = check_head_pose(face_landmarks.landmark, w)
            
            if avg_ear < EAR_THRESHOLD:
                eye_counter += 1
                if eye_counter >= DROWSY_FRAME_LIMIT:
                    drowsy_alert = True
            else:
                eye_counter = 0
                
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= YAWN_FRAME_LIMIT:
                    yawn_alert = True
            else:
                yawn_counter = 0
                
            if is_distracted:
                distraction_counter += 1
                if distraction_counter >= DISTRACTION_FRAME_LIMIT:
                    distraction_alert = True
            else:
                distraction_counter = 0

            # UI Text Displays
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Gaze: {gaze_status}", (30, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Trigger alarms and visual warnings
    if drowsy_alert or yawn_alert or distraction_alert:
        trigger_alarm(True) 
        if drowsy_alert:
            cv2.putText(frame, "DROWSY!", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        if yawn_alert:
            cv2.putText(frame, "YAWNING!", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 140, 255), 3)
        if distraction_alert:
            cv2.putText(frame, "DISTRACTED: LOOK FORWARD!", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
    else:
        trigger_alarm(False)

    cv2.imshow('Video', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

trigger_alarm(False)
cap.release()
cv2.destroyAllWindows()