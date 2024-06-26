import numpy as np
from scipy.integrate import solve_ivp
import cv2

def double_pendulum_equations(t, y, L1, L2, m1, m2):
    theta1, z1, theta2, z2 = y
    c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)

    theta1dot = z1
    theta2dot = z2

    z1dot = (m2 * 9.81 * np.sin(theta2) * c - m2 * s * (L1 * z1**2 * c + L2 * z2**2) -
             (m1 + m2) * 9.81 * np.sin(theta1)) / L1 / (m1 + m2 * s**2)
    z2dot = ((m1 + m2) * (L1 * z1**2 * s - 9.81 * np.sin(theta2) + 9.81 * np.sin(theta1) * c) +
             m2 * L2 * z2**2 * s * c) / L2 / (m1 + m2 * s**2)

    return theta1dot, z1dot, theta2dot, z2dot

def create_double_pendulum_simulation(L1, L2, m1, m2, y0, t_span, t_eval):
    sol = solve_ivp(double_pendulum_equations, t_span, y0, t_eval=t_eval, args=(L1, L2, m1, m2), rtol=1e-10, atol=1e-10)
    return sol.y

def plot_double_pendulum(L1, L2, theta1, theta2):
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)
    return x1, y1, x2, y2

import cv2
import numpy as np

def process_gif(file_name):
    cap = cv2.VideoCapture(file_name)
    x_coords, y_coords = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 150, 0])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    x_coords.append(cX)
                    y_coords.append(cY)
                    cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return x_coords, y_coords