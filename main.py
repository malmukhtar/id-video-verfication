import dlib
import cv2
import imageio
from PIL import Image
import numpy as np
import os
import sys
from pathlib import Path




__author__          = 'Mohamed Almukhtar'
__version__         = '1.0.0'
__status__          = "Development"

def main():
    argv = sys.argv[1:]
    if len(argv) < 2:
        print(f"Please use the following formate to call the script\n\npython verify.py path_to_ID path_to_video(mp4) optional Tolorance")
    else:
        # checking the formate of the image
        if not argv[0].lower().endswith("jpg"):
            img = Image.open(argv[0])
            # converting to jpg
            rgbImg = img.convert("RGB")
            rgbImg.save(f"{os.path.dirname(argv[0])}//{Path(argv[0]).stem}.jpg")
            print(f"Converting image and saving to:\t{os.path.dirname(argv[0])}//{Path(argv[0]).stem}.jpg")
            argv[0] = f"{os.path.dirname(argv[0])}//{Path(argv[0]).stem}.jpg"
            

        #getting the id encoding
        id = getFaceEncodings(argv[0])

        # Working with the video
        if argv[1].lower().endswith("mp4"):
            # Read the video
            video = cv2.VideoCapture(argv[1])

            # frame count
            currentframe = 0
            
            while(True):
                
                # reading from frame
                ret,frame = video.read()
            
                if ret:
                    # if video is still left continue creating images
                    name = 'frame' + str(currentframe) + '.jpg'
                    print (f'Analysing...{name}')
                    # writing the extracted images
                    cv2.imwrite(name, frame)

                    img = getFaceEncodings(name)

                    if compareFaceEncodings(img, id):
                        print("Match found!")
                        break
                    currentframe += 1
                else:
                    print("No Match found!")
                    break

            # cleaning 
            if os.path.isfile(name):
                os.remove(name)

            # Release all space and windows once done
            video.release()
            cv2.destroyAllWindows()
        else:
            print("Unsupported video format")
        
        
def getFaceEncodings(path_to_image):
    # Load image using scipy
    image = imageio.v2.imread(path_to_image)

    # Convert to gray
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # Detect faces using the face detector
    detectedFaces = faceDetector(gray, 1)

    # Get pose/landmarks of those faces
    # Will be used as an input to the function that computes face encodings
    # This allows the neural network to be able to produce similar numbers for faces of the same people, regardless of camera angle and/or face positioning in the image
    faces = [shapePredictor(gray, face) for face in detectedFaces]
    
    # For every face detected, compute the face encodings
    return([np.array(faceRecognitionModel.compute_face_descriptor(image, face, 1)) for face in faces])


def compareFaceEncodings(id, face):
    # Finds the difference between each known face and the given face (that we are comparing)
    return (np.linalg.norm(np.array(id) - np.array(face), axis=1) <= TOLERANCE)




argv = sys.argv[1:]

if __name__ == "__main__":

    # Start point
    
    # Get Face Detector from dlib
    faceDetector = dlib.get_frontal_face_detector()

    # Get Predictor from dlib
    shapePredictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Get the face recognition model (For face encoding/ features))
    faceRecognitionModel = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

    # This is the tolerance for face comparisons and  0.5-0.6 works well
   
    try:
        if argv[2]:
            TOLERANCE = float(argv[2])
    except:
        TOLERANCE = 0.6
    
    # Calling the main function
    main()
