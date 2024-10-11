import cv2

subject = "sub-99"  # subject id


video_input = f"/home/max/data/eeg/raw/{subject}_block_1.asf"  # if webcam==None, use this variable as video input
cap = cv2.VideoCapture(video_input)
last_frame = None
frame_count = 0

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while True:
	print(frame_count)
	ret, frame = cap.read()
	if not ret:
		print("Can't receive frame. Exiting ...")
		break
	print(frame.shape)
	last_frame = frame
	frame_count += 1

cv2.imshow("Last Frame", last_frame)
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
