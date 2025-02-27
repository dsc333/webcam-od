'''
Utility that captures a single frame from the computer's webcam using 
OpenCV and performs object detection using the Google Vision API.  
Frame is saved to local directory. 

NOTE:
Before running, include your GCP API key in .env
'''
import cv2
from datetime import datetime
from google.cloud import vision
from dotenv import load_dotenv
import os
import time

load_dotenv()

# Read API KEY
API_KEY=os.environ.get('API_KEY')

if not API_KEY:
    print('No API key found.')

# Authenticate and connect to client 
client = vision.ImageAnnotatorClient(client_options={"api_key": API_KEY})

# Initialize the camera capture object, 0 is usually the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    raise IOError("Cannot open webcam.")

# 500-ms delay ensures that camera has been initialized.
time.sleep(0.5)

# Read a frame from the camera
ret, frame = cap.read()
if not ret:
    raise IOError("Cannot capture frame.")

# Get timestamp
print(datetime.now())
filename = f'cap_{datetime.now()}.jpg'
cv2.imwrite(filename, frame)
print(f'Frame captured and saved to {filename}.')

with open(filename, "rb") as image_file:
    content = image_file.read()
image = vision.Image(content=content)

response={}
# Object detection
objects = client.object_localization(image=image).localized_object_annotations
print(f"Number of objects found: {len(objects)}")
response['objects'] = {}
for idx, object_ in enumerate(objects):
    response['objects'][f"{object_.name}_{idx}"] = f"{object_.score:0.2f}"
    print(f"{object_.name} (confidence: {object_.score:0.2f})")

# Release the capture 
cap.release()
