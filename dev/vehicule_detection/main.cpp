#include <stdio.h>
#include <iostream>
#include <sstream>
#include <ctype.h>

#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/video.hpp"


#include <fstream>


#include "histogram.h"
#include "colorhistogram.h"
#include "objectFinder.h"

#define IMG_FILENAME "/home/audric/ownCloud/Documents/UTBM/GI/GI05/IN54/Projet/A2013_ProjetIN5x_Data1/imgD/W_3700R.tif"
#define SELECTION_FILE "/home/audric/IN52/selection.txt"
#define REF_FILENAME "E:/DropBox/UTBM/IN52/ref.tif"


using namespace cv;
using namespace std;

/* Pour le camShift */
Mat image;
bool backprojMode = false;
bool selectObject = false;
int trackObject = 0;
bool showHist = true;
Point origin;
Rect selection;
int vmin = 100, vmax = 256, smin = 75;

/*---------------------*/

/* Matching Methode */
Mat img; Mat templ; Mat result;
string image_window = "Source Image";

int match_method=3;
int max_Trackbar = 5;
/*---------------------*/

Mat fgMaskMOG;
Mat fgMaskMOG2;
Ptr<BackgroundSubtractorMOG> pMOG;
Ptr<BackgroundSubtractorMOG2> pMOG2;
IplConvKernel *kernel;
int keyboard;

void processImages(char* firstFrameFilename);
int testCamShift();
Rect getTrackingZoneFromFile(string filename);

int histogramEqua();
void MatchingMethod(int, void*);
int templateMatching();


int main()
{
    pMOG = new BackgroundSubtractorMOG();
    pMOG2 = new BackgroundSubtractorMOG2();


    selection = getTrackingZoneFromFile(SELECTION_FILE);
    cout << "tracking from: " << selection << endl << endl;

    testCamShift();

//    templateMatching();
//    histogramEqua();
//    testCamShift();


//    processImages("IMG_FILENAME");
    destroyAllWindows();
    return EXIT_SUCCESS;
}

int templateMatching()
{
    string fn("IMG_FILENAME") ;

    img = imread( fn, 1 );
    templ = imread(REF_FILENAME, 1 );

    /// Create windows
    namedWindow( image_window,  CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL  );

    int count=3700;
    size_t index = fn.find_last_of("/");
    if(index == string::npos)
        index = fn.find_last_of("\\");

    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    stringstream ss;
    ss << count;
    string nextFrameFilename = prefix + "W_" + ss.str() + "R" + suffix;

    for(;;)
    {
        MatchingMethod(0,0);
        img = imread(nextFrameFilename);
        if(img.empty())
            break;
        count++;
        stringstream ss;
        ss << count;
        nextFrameFilename = prefix + "W_" + ss.str() + "R" + suffix;
        waitKey(10);
    }
    waitKey(0);
    return 0;
}

void MatchingMethod( int, void* )
{
  /// Source image to display
  Mat img_display;
  img.copyTo( img_display );

  /// Create the result matrix
  int result_cols =  img.cols - templ.cols + 1;
  int result_rows = img.rows - templ.rows + 1;

  result.create( result_cols, result_rows, CV_32FC1 );

  /// Do the Matching and Normalize
  matchTemplate( img, templ, result, match_method );
  normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

  /// Localizing the best match with minMaxLoc
  double minVal; double maxVal; Point minLoc; Point maxLoc;
  Point matchLoc;

  minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

  /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
  if( match_method  == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED )
    { matchLoc = minLoc; }
  else
    { matchLoc = maxLoc; }

  /// Show me what you got
  rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );

  imshow( image_window, img_display );

  return;
}

int histogramEqua()
{
    Mat src, dst;

    string source_window = "Source image";
    string equalized_window = "Equalized Image";

    src = imread( "E:/DropBox/UTBM/IN52/imgD/W_3899R.tif", 1 );

    cvtColor( src, src, CV_BGR2GRAY );

    equalizeHist( src, dst );

    namedWindow( source_window, CV_WINDOW_AUTOSIZE );
    namedWindow( equalized_window, CV_WINDOW_AUTOSIZE );

    imshow( source_window, src );
    imshow( equalized_window, dst );

    waitKey(0);

    return 0;
}



int testCamShift()
{
    Rect trackWindow;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

    namedWindow( "CamShift Demo",  CV_WINDOW_AUTOSIZE | CV_GUI_NORMAL  );

    /*createTrackbar( "Vmin", "CamShift Demo", &vmin, 256, 0 );
    createTrackbar( "Vmax", "CamShift Demo", &vmax, 256, 0 );
    createTrackbar( "Smin", "CamShift Demo", &smin, 256, 0 );*/


    Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    bool paused = false;

    string fn =  IMG_FILENAME;


    frame = imread(fn);

    int count=3700;
    size_t index = fn.find_last_of("/");
    if(index == string::npos)
        index = fn.find_last_of("\\");

    size_t index2 = fn.find_last_of(".");
    string prefix = fn.substr(0,index+1);
    string suffix = fn.substr(index2);
    stringstream ss;
    ss << count;
    string nextFrameFilename = prefix + "W_" + ss.str() + "R" + suffix;


    while(true)
    {

        if( !paused )
        {
            frame = imread(nextFrameFilename);
            count++;
            if(frame.empty()) {
                count = 3700;
            }

            stringstream ss;
            ss << count;
            nextFrameFilename = prefix + "W_" + ss.str() + "R" + suffix;
            if(count == 3700) {
                frame = imread(nextFrameFilename);
            }

        }

        frame.copyTo(image);

        if( !paused )
        {
            cvtColor(image, hsv, CV_BGR2HSV);

            if( trackObject )
            {
                int _vmin = vmin, _vmax = vmax;

                inRange(hsv, Scalar(0, smin, MIN(_vmin,_vmax)), Scalar(180, 256, MAX(_vmin, _vmax)), mask);
                int ch[] = {0, 0};
                hue.create(hsv.size(), hsv.depth());
                mixChannels(&hsv, 1, &hue, 1, ch, 1);

                if( trackObject < 0 )
                {
                    Mat roi(hue, selection), maskroi(mask, selection);
                    calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
                    normalize(hist, hist, 0, 255, CV_MINMAX);

                    trackWindow = selection;
                    trackObject = 1;

                    histimg = Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    Mat buf(1, hsize, CV_8UC3);
                    for( int i = 0; i < hsize; i++ )
                        buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
                    cvtColor(buf, buf, CV_HSV2BGR);

                    for( int i = 0; i < hsize; i++ )
                    {
                        int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
                        rectangle( histimg, Point(i*binW,histimg.rows),
                                   Point((i+1)*binW,histimg.rows - val),
                                   Scalar(buf.at<Vec3b>(i)), -1, 8 );
                    }
                }

                calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
                backproj &= mask;
                RotatedRect trackBox = CamShift(backproj, trackWindow,
                                    TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
                if( trackWindow.area() <= 1 )
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
                    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  Rect(0, 0, cols, rows);
                }

                if( backprojMode )
                    cvtColor( backproj, image, CV_GRAY2BGR );
                ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
            }
        }
        else if( trackObject < 0 )
            paused = false;

        if( selectObject && selection.width > 0 && selection.height > 0 )
        {
            Mat roi(image, selection);
            bitwise_not(roi, roi);
        }

        imshow( "CamShift Demo", image );

        char c = (char)waitKey(10);
        if( c == 27 || c=='q')
            break;
        switch(c)
        {
        case 'b':
            backprojMode = !backprojMode;
            break;
        case 'c':
            trackObject = 0;
            histimg = Scalar::all(0);
            break;
        case 'p':
            paused = !paused;
            break;
        default:
            ;
        }
    }

    return 0;
}


void processImages(char* fistFrameFilename) {
    //Traitement de l'image avec la méthode BgSubtractorMOG et bgSubtractorMOG2
    int count=3701;
    Mat frame;
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


/**
 * @brief getTrackingZoneFromFile Format du fichier : x y width height sans saut de ligne
 * @param filename
 * @return
 */
Rect getTrackingZoneFromFile(string filename) {

    std::ifstream infile(filename.c_str());
    int x, y, w, h;
    infile >> w >> h >> x >> y;
    Rect zone;
    zone.x = x;
    zone.y = y;
    zone.width = w;
    zone.height = h;
    trackObject = -1;
    return zone;
}
