from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2


CLASSES = ["background", "aeroplane", "bicycle", "bird"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromTorch('best.pt')
# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
# time.sleep(2.0)
# fps = FPS().start()


# while True:
#     	# grab the frame from the threaded video stream and resize it
# 	# to have a maximum width of 400 pixels
# 	frame = vs.read()
# 	frame = imutils.resize(frame, width=416)
# 	# grab the frame dimensions and convert it to a blob
# 	(h, w) = frame.shape[:2]
# 	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (416, 416)),
# 		0.007843, (416, 416), 127.5)
# 	# pass the blob through the network and obtain the detections and
# 	# predictions
# 	net.setInput(blob)
# 	detections = net.forward()