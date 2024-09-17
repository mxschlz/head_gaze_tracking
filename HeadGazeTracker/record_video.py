import cv2 as cv

def record_video(source=0, output_file="output"):
	# Initialize the video capture object
	cap = cv.VideoCapture(source)  # 0 usually represents the default webcam

	# Check if the webcam is opened successfully
	if not cap.isOpened():
		print("Error: Could not open webcam.")
		exit()

	# Define the codec and create VideoWriter object
	fourcc = cv.VideoWriter_fourcc(*'XVID')  # You can use other codecs if needed
	frame_width = int(cap.get(3))  # Get the width of frames from the webcam
	frame_height = int(cap.get(4))  # Get the height of frames from the webcam
	out = cv.VideoWriter(f'{output_file}.avi', fourcc, 20.0, (frame_width, frame_height))

	# Record video until 'q' is pressed
	while True:
		ret, frame = cap.read()  # Capture frame-by-frame
		if not ret:
			print("Error: Could not read frame.")
			break

		out.write(frame)  # Write the frame to the output video file
		cv.imshow('frame', frame)  # Display the resulting frame

		if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
			break

	# Release everything when done
	cap.release()
	out.release()
	cv.destroyAllWindows()

if __name__ == "__main__":
	record_video()
