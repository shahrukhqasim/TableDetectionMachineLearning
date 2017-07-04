//
// Created by srq on 7/4/17.
//

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
#include "helpers.h"

using namespace std;
using namespace cv;


int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "Oops in arguments!\n";
        return -1;
    }

    string outJsonPath = argv[1];
    string imagePath = argv[2];
    string outputImagePath1 = argv[3];
    string outputImagePath2 = argv[4];

    Mat imageForGt = imread(imagePath, 1);
    Mat imageForOutput = imageForGt.clone();

    ifstream jsonInStream(outJsonPath);
    Json::Value jsonData;
    jsonInStream >> jsonData;

    for (int i = 0; i < jsonData.size(); i++) {
        Json::Value wordData = jsonData[i];
        
        int left = stoi(wordData["left"].asString());
        int top = stoi(wordData["top"].asString());
        int right = stoi(wordData["right"].asString());
        int bottom = stoi(wordData["bottom"].asString());
        bool isTable = wordData["is_table"].asBool();
        bool isTableGt = wordData["is_table_gt"].asBool();

        Rect rect(left, top, right - left, bottom - top);

        rectangle(imageForGt, rect, isTableGt ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 3, 8);
        rectangle(imageForOutput, rect, isTable ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 3, 8);
    }

    imwrite(outputImagePath1, imageForGt);
    imwrite(outputImagePath2, imageForOutput);

    return 0;
}