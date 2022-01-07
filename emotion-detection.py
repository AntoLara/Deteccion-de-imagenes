import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process

imgpath="/home/antonio/Documents/Git/Deteccion-de-imagenes/jose.jpg"
#imgpath = "/home/hugo/Pictures/capture.jpg"

#analyze = DeepFace.analyze(imgpath,actions=['emotion', 'age', 'gender', 'race'],models={}, enforce_detection=True)
analyze = DeepFace.analyze(imgpath, actions=['emotion', 'age', 'gender', 'race'], models={}, enforce_detection=False)

print(analyze)