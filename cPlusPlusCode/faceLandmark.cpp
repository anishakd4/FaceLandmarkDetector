#include<dlib/opencv.h>
#include<dlib/image_processing.h>
#include<dlib/image_processing/frontal_face_detector.h>
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace dlib;
using namespace std;
using namespace cv;


void drawFaceLandmarks(Mat &image, full_object_detection faceLandmark){

    //Loop over all face landmarks
    for(int i=0; i< faceLandmark.num_parts(); i++){
        int x = faceLandmark.part(i).x();
        int y = faceLandmark.part(i).y();
        string text = to_string(i+1);

        //Draw a small circle at face landmark over the image using opencv
        circle(image, Point(x, y), 1, Scalar(0, 0, 255), 2, LINE_AA );

        //Draw text at face landmark to show index of current face landmark over the image using opencv
        putText(image, text, Point(x, y),FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 0, 0), 1);
    }
}

void writeFaceLandmarkstoFile(string facelandmarkFileName, full_object_detection faceLandmark){
    //Open file
    std::ofstream outputFile;
    outputFile.open(facelandmarkFileName);

    //Loop over all face landmarks
    for(int i=0; i<faceLandmark.num_parts(); i++){
        outputFile<<faceLandmark.part(i).x()<<" "<<faceLandmark.part(i).y()<<endl;
    }

    //close file
    outputFile.close();
}

int main(){
    //Loaad the dlib face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    //Load dlib the face landmark detector
    shape_predictor faceLandmarkDetector ;
    deserialize("../../dlibAndModel/shape_predictor_68_face_landmarks.dat") >> faceLandmarkDetector;

    // Read image using opencv
    Mat inputImage = imread("../assets/putin.jpg");

    //use the below image to see how program works for images having multiple faces
    //Mat inputImage = imread("../assets/putinTrumph.jpg");

    //Check if the mnetioned image exits
    if(inputImage.empty()){
        cout<<"image doesn't exist"<<endl;
        return -1;
    }

    //create a copy of the input image to work on so that finally we have input and output images to compare
    Mat image = inputImage.clone();

    //Convert loaded opencv image to dlib image format
    cv_image<bgr_pixel> dlibImage(image);

    //Detect faces in the image and print the number of faces detected
    std::vector<dlib::rectangle> faces = faceDetector(dlibImage);
    cout<<"Number of faces detected:"<<faces.size()<<endl;

    //Get Face landmarks of all detected faces
    std::vector<full_object_detection> facelandmarks;
    for(int i=0; i<faces.size(); i++){

        //Get the face landmark and print number of landmarks detected
        full_object_detection facelandmark = faceLandmarkDetector(dlibImage, faces[i]);
        cout<<"Number of face landmarks detected:"<<facelandmark.num_parts()<<endl;

        //Push face landmark to array of All face's landmarks array
        facelandmarks.push_back(facelandmark);

        //Draw landmarks on image
        drawFaceLandmarks(image, facelandmark);

        //Write face Landmarks to a file on disk to analyze
        string s1= "face_landmarks_";
        string s2 = to_string(i+1);
        string s3 = ".txt";
        string landmarksFilename = s1 + s2 + s3;
        writeFaceLandmarkstoFile(landmarksFilename, facelandmark);
    }

    //Save image with face landmarks drawn to disk to analyze using opencv
    string imageFileName = "image_face_landmark.jpg";
    cv::imwrite(imageFileName,image);

    //Create windows to show images
    namedWindow("Input image", WINDOW_NORMAL);
    namedWindow("Output image", WINDOW_NORMAL);

    //Display both images with face landmarks and without face landmarks being drawn
    imshow("Input image", inputImage);
    imshow("Output image", image);

    //It will pause the program until you press any key from keyboard So that you can see output.
    waitKey(0);

    return 0;
}