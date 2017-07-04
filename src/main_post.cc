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

vector<bool> majorityCountPostProcess(vector<Rect> wordBoxes, vector<bool> areTables) {
    vector<bool> areTablesPost;
    for (int ii = 0; ii < wordBoxes.size(); ii++) {
        Rect i = wordBoxes[ii];
        int count = 0;

        Rect left;
        Rect right;
        Rect upper;
        Rect lower;

        int leftIndex;
        int rightIndex;
        int upperIndex;
        int lowerIndex;

        bool upperFound = false;
        bool lowerFound = false;
        bool leftFound = false;
        bool rightFound = false;


        for (int jj = 0; jj < wordBoxes.size(); jj++) {
            Rect j = wordBoxes[jj];
            if (j == i) {
                continue;
            }

            // Upper
            if( (verticalOverlap(i, j) && (i.y - j.y - j.height > 0)) ) {
                int oldDistance = i.y - upper.y - upper.height;
                int newDistance = i.y - j.y - j.height;

                if (!upperFound || newDistance < oldDistance) {
                    upperFound = true;
                    upper = j;
                    upperIndex = jj;
                }
            }


            // Lower
            if( (verticalOverlap(i, j) && (j.y - i.y - i.height > 0)) ) {
                int oldDistance = lower.y - i.y - i.height;
                int newDistance = j.y - i.y - i.height;

                if (!lowerFound || newDistance < oldDistance) {
                    lowerFound = true;
                    lower = j;
                    lowerIndex = jj;
                }
            }

            // Left
            if( (horizontalOverlap(i, j) && (i.x - j.x - j.width > 0)) ) {
                int oldDistance = i.x - left.x - left.width;
                int newDistance = i.x - j.x - j.width;

                if (!leftFound || newDistance < oldDistance) {
                    leftFound = true;
                    left = j;
                    leftIndex = jj;
                }
            }


            // Right
            if( (horizontalOverlap(i, j) && (j.x - i.x - i.width > 0)) ) {
                int oldDistance = right.x - i.x - i.width;
                int newDistance = j.x - i.x - i.width;

                if (!rightFound || newDistance < oldDistance) {
                    rightFound = true;
                    right = j;
                    rightIndex = jj;
                }
            }
        }

        int tableCount = 0;
        int nonTableCount = 0;

        if(rightFound) {
            if(areTables[rightIndex])
                tableCount++;
            else
                nonTableCount++;
        }
        if(leftFound) {
            if(areTables[leftIndex])
                tableCount++;
            else
                nonTableCount++;
        }
//        if(upperFound) {
//            if(areTables[upperIndex])
//                tableCount++;
//            else
//                nonTableCount++;
//        }
//        if(lowerFound) {
//            if(areTables[lowerIndex])
//                tableCount++;
//            else
//                nonTableCount++;
//        }

        areTablesPost.push_back(tableCount > nonTableCount);
    }

    return areTablesPost;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        cout << "Given "<< argc<<"; expected 6"<<endl;
        cout << "Oops in arguments!\n";
        return -1;
    }

    string outJsonPath = argv[1];
    string imagePath = argv[2];
    string outputImagePath1 = argv[3];
    string outputImagePath2 = argv[4];
    string outputImagePath3 = argv[5];

    Mat imageForGt = imread(imagePath, 1);
    Mat imageForOutput = imageForGt.clone();
    Mat imageForPost = imageForGt.clone();

    ifstream jsonInStream(outJsonPath);
    Json::Value jsonData;
    jsonInStream >> jsonData;

    vector<Rect> rectangles;
    vector<bool> areTables;

    for (int i = 0; i < jsonData.size(); i++) {
        Json::Value wordData = jsonData[i];
        
        int left = stoi(wordData["left"].asString());
        int top = stoi(wordData["top"].asString());
        int right = stoi(wordData["right"].asString());
        int bottom = stoi(wordData["bottom"].asString());
        bool isTable = wordData["is_table"].asBool();
        bool isTableGt = wordData["is_table_gt"].asBool();

        Rect rect(left, top, right - left, bottom - top);

        rectangles.push_back(rect);
        areTables.push_back(isTable);

        rectangle(imageForGt, rect, isTableGt ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 3, 8);
        rectangle(imageForOutput, rect, isTable ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 3, 8);
    }

    vector<bool> areTablesPost = majorityCountPostProcess(rectangles, areTables);

    for (int i = 0; i < rectangles.size(); i++) {
        rectangle(imageForPost, rectangles[i], areTablesPost[i] ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 3, 8);
    }

    imwrite(outputImagePath1, imageForGt);
    imwrite(outputImagePath2, imageForOutput);
    imwrite(outputImagePath3, imageForPost);

    return 0;
}