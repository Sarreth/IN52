#include <stdio.h>
#include <iostream>
#include <sstream>

#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/video.hpp"

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

int main(int argc, char* argv[])
{
    //Creation des fenetres d'affichage de resultats
    namedWindow("Frame");
    namedWindow("FG Mask MOG");

    pMOG = new BackgroundSubtractorMOG();
    pMOG2 = new BackgroundSubtractorMOG2();

    processImages("E:/DropBox/UTBM/IN52/imgD/W_3700R.tif");
    destroyAllWindows();
    return EXIT_SUCCESS;
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


        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
            cv::Scalar(255,255,255), -1);

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
