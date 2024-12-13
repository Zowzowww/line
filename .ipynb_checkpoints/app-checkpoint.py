from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
from PoseModule import PoseDetector

app = Flask(__name__)

# Initialize video capture and pose detector
cap = cv2.VideoCapture(0) #0 only camera #1 two camera
detector = PoseDetector()
count_squat = 0
count_crunch = 0
dir_squat = 0
dir_crunch = 0
pTime = 0

def generate_frames():
    global count_squat, count_crunch, dir_squat, dir_crunch, pTime
    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            img = cv2.resize(img, (1280, 720))
            img = detector.find_pose(img, False)
            lmList = detector.find_position(img, False)
            
            if len(lmList) != 0:
                # Squat detection
                angle_right_leg = detector.find_angle(img, 24, 26, 28)
                angle_left_leg = detector.find_angle(img, 23, 25, 27)
                
                per_squat = np.interp(angle_right_leg, (70, 90), (0, 100))
                bar_squat = np.interp(angle_right_leg, (70, 90), (650, 100))
                color_squat = (255, 0, 255)

                if per_squat == 100 and dir_squat == 0:
                    count_squat += 0.5
                    dir_squat = 1
                if per_squat == 0 and dir_squat == 1:
                    count_squat += 0.5
                    dir_squat = 0

                # Visualize squat
                cv2.rectangle(img, (1100, 100), (1175, 650), color_squat, 3)
                cv2.rectangle(img, (1100, int(bar_squat)), (1175, 650), color_squat, cv2.FILLED)
                cv2.putText(img, f'{int(per_squat)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color_squat, 4)
                cv2.putText(img, f'Squats: {int(count_squat)}', (45, 620), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

                # Crunch detection
                angle_right_crunch = detector.find_angle(img, 12, 24, 26)
                angle_left_crunch = detector.find_angle(img, 11, 23, 25)
                
                per_crunch = np.interp(angle_right_crunch, (45, 90), (0, 100))
                bar_crunch = np.interp(angle_right_crunch, (45, 90), (650, 100))
                color_crunch = (255, 0, 255)

                if per_crunch == 100 and dir_crunch == 0:
                    count_crunch += 0.5
                    dir_crunch = 1
                if per_crunch == 0 and dir_crunch == 1:
                    count_crunch += 0.5
                    dir_crunch = 0

                # Visualize crunch
                cv2.rectangle(img, (50, 100), (125, 650), color_crunch, 3)
                cv2.rectangle(img, (50, int(bar_crunch)), (125, 650), color_crunch, cv2.FILLED)
                cv2.putText(img, f'{int(per_crunch)} %', (50, 75), cv2.FONT_HERSHEY_PLAIN, 4, color_crunch, 4)
                cv2.putText(img, f'Crunches: {int(count_crunch)}', (45, 670), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 5)

            # Frame rate display
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html') #connection templates index.html

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)