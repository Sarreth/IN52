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
#define REF_FILENAME "/home/audric/IN52/ref.tif"

#define WINDOW_NAME "Projet d'IN54"
#define RESIZE_VAL 4


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
Mat imgOri;
string image_window = "Source Image";

int match_method=3;
int max_Trackbar = 5;
/*---------------------*/


void processImages(char* firstFrameFilename);
int testCamShift();
Rect getTrackingZoneFromFile(char* filename);

int histogramEqua();
void MatchingMethod(int, void*);
int templateMatching();
Mat DFF(string path);


int main()
{
    selection = getTrackingZoneFromFile(SELECTION_FILE);
    cout << "tracking from: " << selection << endl << endl;

    testCamShift();

//    templateMatching();
//    histogramEqua();
//    testCamShift();


    destroyAllWindows();
    return EXIT_SUCCESS;
}

int templateMatching()
{
    string fn(IMG_FILENAME);
    imgOri = imread( fn, 1 );

    templ = DFF(REF_FILENAME);
    img = DFF(IMG_FILENAME);

    /// Create windows
    namedWindow( image_window, CV_WINDOW_AUTOSIZE );

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
        imgOri = imread(nextFrameFilename);
        if(imgOri.empty())
            break;

        img = DFF(nextFrameFilename);
        count++;
        stringstream ss;
        ss << count;
        nextFrameFilename = prefix + "W_" + ss.str() + "R" + suffix;
        char c = (char)waitKey(5);
        if( c == 27 )
            break;
    }
    waitKey(0);
    return 0;
}

Mat DFF(string path)
{
    Mat I = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    Size size = I.size();
    size.height /= RESIZE_VAL;
    size.width /= RESIZE_VAL;
    resize(I,I,size);

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    return I;
}

void MatchingMethod( int, void* )
{
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
     rectangle( imgOri, matchLoc*RESIZE_VAL, Point( matchLoc.x*RESIZE_VAL + templ.cols*RESIZE_VAL , matchLoc.y*RESIZE_VAL + templ.rows*RESIZE_VAL ), Scalar(0,0,255), 2, 8, 0 );

     imshow( image_window, imgOri );

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

    namedWindow( WINDOW_NAME, CV_WINDOW_AUTOSIZE );

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


    for(;;)
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

        imshow( WINDOW_NAME, image );

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


void processImages(char* fistFrameFilename)
{

}


/**
 * @brief getTrackingZoneFromFile Format du fichier : x y width height sans saut de ligne
 * @param filename
 * @return
 */
Rect getTrackingZoneFromFile(char* filename) {

    std::ifstream infile(filename);
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
