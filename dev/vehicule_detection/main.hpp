#ifndef MAIN_HPP
#define MAIN_HPP

#include "opencv/cv.h"

using namespace cv;
using namespace std;


void processImages(char* firstFrameFilename);
int testCamShift(Rect selection);
Rect getTrackingZoneFromFile(string filename);
void getSuffixAndPrefix(string fn, string & suffix, string & prefix);
string getImageFilename(string prefix, int count, string suffix);

int histogramEqua();
int MatchingMethod(int, void*, string, Mat&);
Mat DFF(string path);
int templateMatching();




#endif // MAIN_HPP
