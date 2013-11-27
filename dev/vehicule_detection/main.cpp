#include <stdio.h>
#include <iostream>
#include <sstream>

#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/video.hpp"

#include "histogram.h"
#include "colorhistogram.h"
#include "objectFinder.h"

using namespace cv;
using namespace std;

Mat frame;
Mat fgMaskMOG;
Mat fgMaskMOG2;
Ptr<BackgroundSubtractorMOG> pMOG;
Ptr<BackgroundSubtractorMOG2> pMOG2;
IplConvKernel *kernel;
int keyboard;

void processVideo(char* videoFilename);
void processImages(char* firstFrameFilename);
int testCamShift();

int main(int argc, char* argv[])
{
    //Creation des fenetres d'affichage de resultats
//    namedWindow("Frame");
//    namedWindow("FG Mask MOG");

    pMOG = new BackgroundSubtractorMOG();
    pMOG2 = new BackgroundSubtractorMOG2();

    testCamShift();

//    processImages("E:/DropBox/UTBM/IN52/imgD/W_3700R.tif");
    destroyAllWindows();
    return EXIT_SUCCESS;
}

int testCamShift()
{
     // Read reference image
     Mat image= imread("E:/DropBox/UTBM/IN52/ref.tif");
     if (!image.data)
         return 0;

     // Define ROI
     Mat imageROI= image(Rect(110,260,35,40));
     rectangle(image, Rect(110,260,35,40),Scalar(0,0,255));

     // Display image
     namedWindow("Image");
     imshow("Image",image);

     // Get the Hue histogram
     int minSat=65;
     ColorHistogram hc;
     //MatND colorhist= hc.getHueHistogram(imageROI,minSat);

     ObjectFinder finder;
     //finder.setHistogram(colorhist);
     finder.setThreshold(0.2f);

     // Convert to HSV space
     Mat hsv;
     cvtColor(image, hsv, CV_BGR2HSV);

     // Split the image
     vector<Mat> v;
     split(hsv,v);

     // Eliminate pixels with low saturation
     threshold(v[1],v[1],minSat,255,THRESH_BINARY);
     namedWindow("Saturation");
     imshow("Saturation",v[1]);

     // Get back-projection of hue histogram
     int ch[1]={0};
     Mat result= finder.find(hsv,0.0f,180.0f,ch,1);

     namedWindow("Result Hue");
     imshow("Result Hue",result);

     bitwise_and(result,v[1],result);
     namedWindow("Result Hue and");
     imshow("Result Hue and",result);

     // Second image
     image= imread("E:/DropBox/UTBM/IN52/imgD/W_3700R.tif");

     // Display image
     namedWindow("Image 2");
     imshow("Image 2",image);

     // Convert to HSV space
     cvtColor(image, hsv, CV_BGR2HSV);

     // Split the image
     split(hsv,v);

     // Eliminate pixels with low saturation
     threshold(v[1],v[1],minSat,255,THRESH_BINARY);
     namedWindow("Saturation");
     imshow("Saturation",v[1]);

     // Get back-projection of hue histogram
     result= finder.find(hsv,0.0f,180.0f,ch,1);

     namedWindow("Result Hue");
     imshow("Result Hue",result);

     // Eliminate low stauration pixels
     bitwise_and(result,v[1],result);
     namedWindow("Result Hue and");
     imshow("Result Hue and",result);

     // Get back-projection of hue histogram
     finder.setThreshold(-1.0f);
     result= finder.find(hsv,0.0f,180.0f,ch,1);
     bitwise_and(result,v[1],result);
     namedWindow("Result Hue and raw");
     imshow("Result Hue and raw",result);

     Rect rect(110,260,35,40);
     rectangle(image, rect, Scalar(0,0,255));

     TermCriteria criteria(TermCriteria::MAX_ITER,10,0.01);
 //  cout << "meanshift= " << meanShift(result,rect,criteria) << endl;

     rectangle(image, rect, Scalar(0,255,0));

     // Display image
     namedWindow("Image 2 result");
     imshow("Image 2 result",image);

     waitKey();
     return 0;
}

void processImages(char* fistFrameFilename) {
    //Traitement de l'image avec la méthode BgSubtractorMOG et bgSubtractorMOG2
    int count=3701;
    frame = imread(fistFrameFilename);

    string fn(fistFrameFilename);

    while( (char)keyboard != 'q' && (char)keyboard != 27 )
    {
        //Operation MOG sur l'image
        pMOG->operator()(frame, fgMaskMOG);
        pMOG2->operator()(frame, fgMaskMOG2);

        kernel = cvCreateStructuringElementEx(5, 5, 2, 2, CV_SHAPE_ELLIPSE);

        IplImage* mask = new IplImage(fgMaskMOG2);
        //Traitement de l'image pour eliminer des parasites
        //si on erode avant de dilate, on voit surtout le feuillage des arbres et la voiture disparait

        cvDilate(mask, mask, kernel, 2);
        cvErode(mask, mask, kernel, 2);

        //avec un erode de 10 on supprime quasiment entièrement la voiture et seul le ciel et les arbres apparaissent
        //peut-être utilisable pour connaitre position des parasites et refaire une soustraction
        //cvErode(mask, mask, kernel, 10);

        //Recuperation du prefix et suffix du fichier
        size_t index = fn.find_last_of("/");
        if(index == string::npos)
            index = fn.find_last_of("\\");

        size_t index2 = fn.find_last_of(".");
        string prefix = fn.substr(0,index+1);
        string suffix = fn.substr(index2);


        rectangle(frame, Point(10, 2), Point(100,20),
            Scalar(255,255,255), -1);

        //Affichage du resultat
        imshow("Frame", frame);
        imshow("FG Mask MOG", fgMaskMOG);
//        imshow("FG Mask MOG 2", fgMaskMOG2);
        cvShowImage("GeckoGeek Mask", mask);

        keyboard = waitKey( 30 );

        //Recuperation nom de la prochaine image
        stringstream ss;
        ss << count;
        string nextFrameFilename = prefix + "W_" + ss.str() + "R" + suffix;
        count++;

        frame = imread(nextFrameFilename);
        if(!frame.data)
        {
            cerr << "Unable to open image frame: " << nextFrameFilename << endl;
            exit(EXIT_FAILURE);
        }
        fn.assign(nextFrameFilename);
    }
}
