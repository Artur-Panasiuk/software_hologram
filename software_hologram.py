import cv2
import mediapipe as mp
import pyvista as pv
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

plotter = pv.Plotter()
mesh = pv.read("test_cube.obj") 
actor = plotter.add_mesh(mesh, color="cyan", smooth_shading=True)
actor.orientation = (0, 0, 0)
plotter.camera_position = 'xy'
plotter.show(auto_close=False, interactive_update=True)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    smoothed_scale = 0.01

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        nose = landmarks[1]

        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        cv2.circle(frame, (nose_x, nose_y), 5, (0, 0, 255), -1)

        dx = (nose_x - w/2) / (w/2)
        offset_y = 0.06
        dy = ((nose_y - offset_y) - h/2) / (h/2)
        yaw = dx * 60
        pitch = -dy * 60
        roll = 0
        actor.orientation = (pitch, yaw, roll)

        z_norm = nose.z
        raw_scale = np.interp(z_norm, [-0.15, 0.05], [8.0, 0.2])  

        alpha = 0.1
        smoothed_scale = (1 - alpha) * smoothed_scale + alpha * raw_scale

        actor.SetScale(smoothed_scale, smoothed_scale, smoothed_scale)

        print(f"Rot -> X:{pitch:.1f}, Y:{yaw:.1f}, Z:{roll:.1f}, Scale:{smoothed_scale:.2f}")

    plotter.update()

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()
plotter.close()
