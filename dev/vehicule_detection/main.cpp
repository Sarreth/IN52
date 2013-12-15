#include "relative_data.hpp"
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
#include "main.hpp"

/* ATTENTION POUR NE PAS CONTINUER A DEVOIR MODIFIER LES IMAGES A CHQUE FOIS, voila la marche à suivre :
créer un nouveau fichier hpp s'appelant relative_data.hpp ET DECOCHER "add to source control : git"
ensuite, copier coller vos define propre à vous uniquement (y en a 3 je crois) et avant de faire un commit
ajouter relative_data.hpp à votre gitignore. Normalement, il le commitra pas, mais on aura chacun nos paramètres comme il faut
*/

#define WINDOW_NAME "Projet d'IN54"
#define RESIZE_VAL 4


using namespace cv;
using namespace std;
/*
Mat fgMaskMOG;
Mat fgMaskMOG2;
Ptr<BackgroundSubtractorMOG> pMOG;
Ptr<BackgroundSubtractorMOG2> pMOG2;
IplConvKernel *kernel;
int keyboard;
*/


int main()
{
    //pMOG = new BackgroundSubtractorMOG();
    //pMOG2 = new BackgroundSubtractorMOG2();

    Rect selection = getTrackingZoneFromFile(SELECTION_FILE);
    //testCamShift(selection);

    templateMatching();
    //    histogramEqua();

    //    processImages("IMG_FILENAME");
    destroyAllWindows();
    return EXIT_SUCCESS;
}

int templateMatching()
{
    Mat img, imgOri, subImage, plaque;
    string image_window = "Source Image";
    string sub_window = "Sub Image";

    string fn(IMG_FILENAME);
    imgOri = imread( fn, 1 );
    img = DFF(IMG_FILENAME,RESIZE_VAL);
    Mat templ = DFF(REF_FILENAME,RESIZE_VAL);
    Mat templ_ima = imread(IMA_FILENAME,1);

    string prefix;
    string suffix;
    namedWindow( sub_window, CV_WINDOW_AUTOSIZE );

    namedWindow( image_window, CV_WINDOW_AUTOSIZE );
    int count=3700;

    getSuffixAndPrefix(fn, suffix, prefix);
    string nextFrameFilename = getImageFilename(prefix, count, suffix);

    while(!(subImage = MatchingMethod(0,0, nextFrameFilename, templ,RESIZE_VAL)).empty())
    {
        plaque = MatchingMethod(0,0, subImage, templ_ima,1);
        count++;
        nextFrameFilename = getImageFilename(prefix, count, suffix);
        imshow( sub_window, plaque );
        char c = (char)waitKey(5);
        if( c == 27 )
            return 0;
    }
    return 0;
}

Mat DFF(string path, int resizeVal)
{
    Mat I = imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    Size size = I.size();
    size.height /= resizeVal;
    size.width /= resizeVal;
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

Mat DFF(Mat img, int resizeVal)
{
    Mat I;
    cvtColor(img, I, CV_BGR2GRAY);
    Size size = I.size();
    size.height /= resizeVal;
    size.width /= resizeVal;
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


Mat MatchingMethod( int, void*, string path, Mat& templ, int resizeVal)
{
    int match_method=3;
    Mat imgOri, img;
    string image_window = "Source Image";
    Mat result;

    imgOri = imread(path);
    if(imgOri.empty())
    {
        Mat empty;
        return empty;
    }
    img = DFF(path,resizeVal);

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
    rectangle( imgOri, matchLoc*resizeVal, Point( matchLoc.x*resizeVal + templ.cols*resizeVal , matchLoc.y*resizeVal + templ.rows*resizeVal ), Scalar(0,0,255), 2, 8, 0 );
    Rect trackRect(matchLoc*resizeVal, Point( matchLoc.x*resizeVal + templ.cols*resizeVal , matchLoc.y*resizeVal + templ.rows*resizeVal ));
    Mat subImage = imgOri(trackRect);
    //imshow( image_window, imgOri );

    return subImage;
}


Mat MatchingMethod( int, void*, Mat imgOri, Mat& templOri, int resizeVal)
{
    int match_method=3;
    Mat img(imgOri),templ(templOri);
    string image_window = "Source Image";
    Mat result;

    if(imgOri.empty())
    {
        Mat empty;
        return empty;
    }

    img = DFF(img,resizeVal);
    templ = DFF(templ,resizeVal);
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
    rectangle( imgOri, matchLoc*resizeVal, Point( matchLoc.x*resizeVal + templ.cols*resizeVal , matchLoc.y*resizeVal + templ.rows*resizeVal ), Scalar(0,0,255), 2, 8, 0 );
    Rect trackRect(matchLoc*resizeVal, Point( matchLoc.x*resizeVal + templ.cols*resizeVal , matchLoc.y*resizeVal + templ.rows*resizeVal ));
    Mat subImage = imgOri(trackRect);
    imshow( image_window, imgOri );

    return subImage;
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



int testCamShift(Rect selection)
{

    Mat image, subimage, frame, hsv,
            hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;

    Rect trackWindow, rectCamshift;

    float ratioCar = ((float)selection.width) /selection.height;

    bool backprojMode = false;
    int vmin = 100, vmax = 256, smin = 75, hsize = 16, count=3700, trackObject = -1;

    float hranges[] = {0,180};
    const float* phranges = hranges;
    string prefix, suffix, nextFrameFilename;

    namedWindow( WINDOW_NAME, CV_WINDOW_AUTOSIZE );


    /*createTrackbar( "Vmin", WINDOW_NAME, &vmin, 256, 0 );
    createTrackbar( "Vmax", WINDOW_NAME, &vmax, 256, 0 );
    createTrackbar( "Smin", WINDOW_NAME, &smin, 256, 0 );*/


    getSuffixAndPrefix(IMG_FILENAME, suffix, prefix);

    cout << " ratioCar :" << ratioCar << endl;

    while(true)
    {

        nextFrameFilename = getImageFilename(prefix, count, suffix);
        frame = imread(nextFrameFilename);

        count++;
        if(frame.empty()) {
            count = 3700;
            nextFrameFilename = getImageFilename(prefix, count, suffix);
            frame = imread(nextFrameFilename);
        }

        frame.copyTo(image);

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


            if( backprojMode )
                cvtColor( backproj, image, CV_GRAY2BGR );
            //ellipse( image, trackBox, Scalar(0,0,255), 3, CV_AA );
            rectCamshift = trackBox.boundingRect();

            float currentRatio =  ( ((float)rectCamshift.width)/rectCamshift.height );
            cout << " currentRatio :" <<currentRatio<< endl;
            rectCamshift.height *= (currentRatio/ratioCar);


            rectangle(image, rectCamshift,  Scalar(0,0,255), 3, CV_AA );

        }

        subimage = image(rectCamshift);
        imshow( WINDOW_NAME, subimage );

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
        default:
            ;
        }
    }


    return 0;
}


void processImages(char* fistFrameFilename)
{
    /*
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
    */
}


void getSuffixAndPrefix(string fn, string & suffix, string & prefix) {
    size_t index = fn.find_last_of("/");
    if(index == string::npos)
        index = fn.find_last_of("\\");

    size_t index2 = fn.find_last_of(".");
    prefix = fn.substr(0,index+1);
    suffix = fn.substr(index2);
}


string getImageFilename(string prefix, int count, string suffix) {
    stringstream ss;
    ss << count;
    string filename = prefix + "W_" + ss.str() + "R" + suffix;
    return filename;
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
    return zone;
}
