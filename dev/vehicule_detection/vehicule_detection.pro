#-------------------------------------------------
#
# Project created by QtCreator 2013-11-14T01:08:23
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = vehicule_detection
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app
INCLUDEPATH +=  E:\OpenCV\opencv_bin\install\include\ \

LIBS += -LE:\OpenCV\opencv_bin\install\x64\mingw\bin \
        -llibopencv_core247d \
        -llibopencv_highgui247d \
        -llibopencv_imgproc247d \
        -llibopencv_features2d247d \
        -llibopencv_calib3d247d \
        -llibopencv_contrib247d \
        -llibopencv_flann247d \
        -llibopencv_gpu247d \
        -llibopencv_ml247d \
        -llibopencv_objdetect247d \
        -llibopencv_legacy247d \
        -llibopencv_video247d

#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/video/video.hpp"

SOURCES += main.cpp \
    objectFinder.cpp \
    histogram.cpp

HEADERS += \
    objectFinder.h \
    histogram.h \
    colorhistogram.h
