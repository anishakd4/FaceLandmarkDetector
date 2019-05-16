import cv2
import dlib
import numpy as np


def drawFaceLandmarks(image, facelandmark):

    #Loop over all face landmarks
    for i, part in enumerate(facelandmark.parts()):
        x = int(part.x)
        y = int(part.y)

        #Draw a small circle at face landmark over the image using opencv
        cv2.circle(image, (x, y), 1, (255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        
        #Draw text at face landmark to show index of current face landmark over the image using opencv
        cv2.putText(image, str(i+1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)


def writeFacelandmarksToDisk(textFileName, faceLandmark):
    with open(textFileName, 'w') as f:
        for i in faceLandmark.parts():
            f.write("%s %s\n" %(int(i.x), int(i.y)))

    f.close()



#Load the  dlib face detector
faceDetector = dlib.get_frontal_face_detector()

#Load the dlib face landmark detector
faceLandmarkDetector = dlib.shape_predictor("../../dlibAndModel/shape_predictor_68_face_landmarks.dat")

# Read image using opencv
inputImage = cv2.imread("../assets/putin.jpg")

#use the below image to see how program works for images having multiple faces
inputImage = cv2.imread("../assets/putinTrumph.jpg")

#create a copy of the input image to work on so that finally we have input and output images to compare
image = np.copy(inputImage)

#convert loaded opencv image to dlib image format
dlibImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Detect faces in the image and print the number of faces detected in the image
faces = faceDetector(dlibImage, 0)
print("Number of faces detected: ", len(faces))

#Get Face landmarks of all detected faces
faceLandmarks = []
for i in range(0, len(faces)):
    dlibFaceRectangle = dlib.rectangle(int(faces[i].left()), int(faces[i].top()), 
        int(faces[i].right()), int(faces[i].bottom()))

    #Get the Face landmarks and print number of face landmarks Detected
    faceLandmark = faceLandmarkDetector(dlibImage, dlibFaceRectangle)
    print("Number of Face landmarks detected: ", len(faceLandmark.parts()))

    #Push face landmark to array of All face's landmarks array
    faceLandmarks.append(faceLandmark)

    #Draw landmarks on image
    drawFaceLandmarks(image, faceLandmark)

    #Write landmarks to a file on disk to analyze
    textFileName = "face_landmarks_" + "_" + str(i+1) + ".txt"
    writeFacelandmarksToDisk(textFileName, faceLandmark)


#Save image with face landmarks drawn to the disk
cv2.imwrite("image_face_landmark.jpg", image)

#Create windows to show images
cv2.namedWindow("Input image", cv2.WINDOW_NORMAL)
cv2.namedWindow("Output image", cv2.WINDOW_NORMAL)

#show input image and output image with face landmarks drawn to visualize
cv2.imshow("Input image", inputImage)
cv2.imshow("Output image", image)

#It will pause the program until you press any key from keyboard So that you can see output.
cv2.waitKey(0)

#Close on the window images opened by this program
cv2.destroyAllWindows()
