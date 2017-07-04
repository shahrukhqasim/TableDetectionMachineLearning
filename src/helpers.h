//
// Created by srq on 7/4/17.
//

#ifndef TABLESMACHINELEARNING_HELPERS_H
#define TABLESMACHINELEARNING_HELPERS_H

#include <iostream>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <docproc/binarize/binarize.h>
#include <docproc/segment/segment.h>
#include <docproc/clean/clean.h>
#include <docproc/utility/utility.h>
#include <json/json/json.h>

bool horizontalOverlap(cv::Rect i, cv::Rect j);
bool verticalOverlap(cv::Rect i, cv::Rect j);

#endif //TABLESMACHINELEARNING_HELPERS_H
