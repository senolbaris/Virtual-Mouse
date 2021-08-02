import cv2
import mediapipe
import autopy
import numpy as np

# Camera Settings
camera_width = 960
camera_height = 540
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, camera_width)
cap.set(4, camera_height)

# Installing mediapipe
mediapipe_hands = mediapipe.solutions.hands
mediapipe_drawing = mediapipe.solutions.drawing_utils
mpHandsLandmarks = mediapipe_hands.Hands()

window = 100

smooth_rate = 11
pre_x, pre_y = 0, 0
cur_x, cur_y = 0, 0

def main(draw_landmarks = True):
	while True:
		success, frame = cap.read()

		imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # mediapipe requires RGB
		results = mpHandsLandmarks.process(imgRGB)
		fingers = []

		cv2.rectangle(frame, (window, window), (camera_width-300, camera_height-window), (255, 0, 255), 2)

		if results.multi_hand_landmarks:
			for landmark in results.multi_hand_landmarks:
				for position, lm in enumerate(landmark.landmark):
					img_height, img_width, img_channel = frame.shape
					x = int(lm.x*img_width)
					y = int(lm.y*img_height)
					fingers.append([position, x, y])

			if draw_landmarks == True:
				mediapipe_drawing.draw_landmarks(frame, landmark, mediapipe_hands.HAND_CONNECTIONS)

			screen_width, screen_height = autopy.screen.size()		

			if fingers[7][2] < fingers[6][2]: #checking if index finger is up if true then move
				large_x = np.interp(fingers[7][1], (window, camera_width-300), (0, screen_width)) # scaling x 
				large_y = np.interp(fingers[7][2], (window, camera_height-100), (0, screen_height)) # scaling y

				cur_x = pre_x + (large_x-pre_x) / smooth_rate # smoothing cursor's move
				cur_y = pre_y + (large_y-pre_y) / smooth_rate
				autopy.mouse.move(screen_width-large_x, large_y)
				cur_x, cur_y = pre_x, pre_y

			if fingers[7][2] < fingers[6][2] and fingers[11][2] < fingers[10][2]: 
				distance = (((fingers[7][1]-fingers[11][1])**2)+((fingers[7][2]-fingers[11][2])**2))**(1/2) # distance between index finger and middle finger
				if distance < 25.0:
					autopy.mouse.click()

		if success:
			cv2.imshow("Virtual Mouse", frame)

		if cv2.waitKey(1) == ord('q'):
			cv2.destroyAllWindows()
			break

try:
	main()
except ValueError:
	pass