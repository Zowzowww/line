import cv2
import mediapipe as mp
import math

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True, enable_seg=False, smooth_seg=True):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity, self.smooth, self.enable_seg, self.smooth_seg)
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True): #
        lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lm_list
        
    def find_angle(self, img, p1, p2, p3, draw=True):
        """
        Calculate the angle between three points.
        
        :param img: Image to draw the angle on
        :param p1: Landmark ID of the first point
        :param p2: Landmark ID of the second point (vertex of the angle)
        :param p3: Landmark ID of the third point
        :param draw: Whether to draw the angle and points on the image
        :return: Calculated angle in degrees
        """
        # Get the coordinates of the three points
        lm_list = self.find_position(img)
        if len(lm_list) < max(p1, p2, p3):
            return None  # Ensure landmarks are valid
        
        x1, y1 = lm_list[p1][1], lm_list[p1][2]
        x2, y2 = lm_list[p2][1], lm_list[p2][2]
        x3, y3 = lm_list[p3][1], lm_list[p3][2]

        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # Draw the points and angle on the image
        if draw:
            cv2.circle(img, (x1, y1), 10, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 0), 2)
            cv2.putText(img, f"{int(angle)}°", (x2 - 50, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return angle

    def predict_pose(self, lmlist, lr_model, nb_model):
        if len(lmlist) == 0:
            return "No landmarks detected", "No landmarks detected"

        # Flatten the list of landmarks for model input
        pose_features = [coord for landmark in lmlist for coord in landmark]

        # Make predictions
        lr_prediction = lr_model.predict([pose_features])[0]  # 0: Squat, 1: Crunch
        nb_prediction = nb_model.predict([pose_features])[0]  # 0: Squat, 1: Crunch

        # Translate predictions to human-readable labels
        lr_result = "Squat" if lr_prediction == 0 else "Crunch"
        nb_result = "Squat" if nb_prediction == 0 else "Crunch"

        return lr_result, nb_result