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

using namespace std;
using namespace cv;


bool horizontalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.y + i.height, j.y + j.height) - std::max(i.y, j.y)) > 0;
}

bool verticalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.x + i.width, j.x + j.width) - std::max(i.x, j.x)) > 0;
}

struct WordData {
    double left;
    double top;
    double right;
    double bottom;
    double width;
    double height;
    double distancePrevX;
    double distanceNextX;
    double distanceAboveY;
    double distanceBelowY;
    bool isTable = false;
};

double scale(int max, int value) {
    return value;//min(1.00,((double)value)/max);
}

vector<WordData> findFeatures(std::vector<std::string> words, std::vector<Rect> wordBoxes, int imageWidth, int imageHeight) {
    vector<WordData> wordFeatures;

    for (int ii = 0; ii < wordBoxes.size(); ii++) {
        Rect i = wordBoxes[ii];
        int count = 0;

        Rect left;
        Rect right;
        Rect upper;
        Rect lower;

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
                }
            }


            // Lower
            if( (verticalOverlap(i, j) && (j.y - i.y - i.height > 0)) ) {
                int oldDistance = lower.y - i.y - i.height;
                int newDistance = j.y - i.y - i.height;

                if (!lowerFound || newDistance < oldDistance) {
                    lowerFound = true;
                    lower = j;
                }
            }

            // Left
            if( (horizontalOverlap(i, j) && (i.x - j.x - j.width > 0)) ) {
                int oldDistance = i.x - left.x - left.width;
                int newDistance = i.x - j.x - j.width;

                if (!leftFound || newDistance < oldDistance) {
                    leftFound = true;
                    left = j;
                }
            }


            // Right
            if( (horizontalOverlap(i, j) && (j.x - i.x - i.width > 0)) ) {
                int oldDistance = right.x - i.x - i.width;
                int newDistance = j.x - i.x - i.width;

                if (!rightFound || newDistance < oldDistance) {
                    rightFound = true;
                    right = j;
                }
            }
        }

        WordData wordData;
        wordData.left = scale(imageWidth, i.x);
        wordData.top = scale(imageHeight, i.y);
        wordData.right = scale(imageWidth, i.x + i.width);
        wordData.bottom = scale(imageHeight, i.y + i.height);
        wordData.width = scale(imageWidth, i.width);
        wordData.height = scale(imageHeight, i.height);
        wordData.distancePrevX = leftFound ? scale(imageWidth, i.x - left.x - left.width) : -1;
        wordData.distanceNextX = rightFound ? scale(imageWidth, right.x - i.x - i.width) : -1;
        wordData.distanceBelowY = lowerFound ? scale(imageHeight, lower.y - i.y - i.height) : -1;
        wordData.distanceAboveY = upperFound ? scale(imageHeight, i.y - upper.y - upper.height) : -1;

        wordFeatures.push_back(wordData);
    }

    return wordFeatures;
}

int main(int argc, char**argv) {
    // Read the image
    if(argc!=4) {
        cout<<"Improper arguments \n";
        return -1;
    }

    string imageFileName = argv[1];
    string ocrFileName = argv[2];
    string gtFileName = argv[3];

    cout<<"Files:"<<endl;
    cout<<imageFileName<<endl;
    cout<<ocrFileName<<endl;
    cout<<gtFileName<<endl<<endl;

    Mat image;
    image = imread(imageFileName, 0);
    Mat imageRgb = imread(imageFileName, 1);

    int imageHeight = image.rows;
    int imageWidth = image.cols;

    // Binarize the image
    Mat binarizedImage;
    Mat temp;
//    docproc::binarize::binarizeBG(image, temp, binarizedImage);


    std::vector<std::string> words;
    std::vector<Rect> bboxes;

    {
        Json::Value json;
        ifstream jsonStream(ocrFileName);
        jsonStream >> json;
        cout<<json.size()<<endl;
        for (int i = 0 ; i < json.size(); i++) {
            Json::Value wordJson = json[i];
//            cout<<wordJson.asString()<<endl;
            string word = wordJson["word"].asString();
            string left = wordJson["left"].asString();
            string top = wordJson["top"].asString();
            string right = wordJson["right"].asString();
            string bottom = wordJson["bottom"].asString();

            cout<<word<<" "<<endl;
            int iLeft = stoi(left);
            int iTop = stoi(top);
            int iRight = stoi(right);
            int iBottom = stoi(bottom);

            // Pdf coordinates
            iBottom = imageHeight - iBottom;
            iTop = imageHeight - iTop;

            words.push_back(word);
            bboxes.push_back(Rect(iLeft,iTop,iRight-iLeft,iBottom-iTop));
//            cout<<"Hello :)" << Rect(iLeft,iTop,iRight-iLeft,iBottom-iTop) <<endl;
        }
    }
    std::vector<Rect> tableBoxes;
    {
        Json::Value json;
        ifstream jsonStream(gtFileName);
        jsonStream >> json;
        cout<<json.size()<<endl;
        for (int i = 0 ; i < json.size(); i++) {
            Json::Value tableJson = json[i];
//            cout<<wordJson.asString()<<endl;
            string left = tableJson["left"].asString();
            string top = tableJson["top"].asString();
            string right = tableJson["right"].asString();
            string bottom = tableJson["bottom"].asString();

            int iLeft = stoi(left);
            int iTop = stoi(top);
            int iRight = stoi(right);
            int iBottom = stoi(bottom);

            // Pdf coordinates
//            iBottom = imageHeight - iBottom;
//            iTop = imageHeight - iTop;

            tableBoxes.push_back(Rect(iLeft,iTop,iRight-iLeft,iBottom-iTop));
            cout<<"Oh its a table"<<endl;
//            cout<<"Hello :)" << Rect(iLeft,iTop,iRight-iLeft,iBottom-iTop) <<endl;
        }
    }
    vector<WordData> wordData = findFeatures(words, bboxes, image.cols, image.rows);

    for (int i = 0; i < words.size(); i++) {
        Rect rect = bboxes[i];
        for (auto j : tableBoxes) {
            if ((rect & j).area() > 0) {
                wordData[i].isTable = 1;
            }
        }
    }
    int ii = 0;
    for (auto i : bboxes) {
//        cout<<"Hello :)" << i <<endl;
        bool isTable = wordData[ii].isTable;
        ii++;

        rectangle(imageRgb, i, isTable ? Scalar(0, 255, 0) : Scalar(0, 0, 255), 3, 8, 0);
    }
    for (auto i : tableBoxes) {
        cout << i << endl;
        rectangle(imageRgb, i, Scalar(255, 0, 0), 3, 8, 0);
    }



    string withoutExtension = docproc::utility::getFileNameWithoutExtension(imageFileName);
    string justName = withoutExtension.substr(withoutExtension.find_last_of('/')+1);
    string fileNameFeatures = withoutExtension + ".json";

    imwrite(withoutExtension+"_rects.png", imageRgb);
    ofstream fileOutFeatures (fileNameFeatures);
    fileOutFeatures << "word,left,top,right,bottom,width,height,previous,next,above,below,is_table"<<endl;


    Json::Value outFileWordData;

    ii = 0;
    for (auto i : wordData) {
        Json::Value singleWordData;
        singleWordData["word"] = words[ii];
        singleWordData["left"] = i.left;
        singleWordData["top"] = i.top;
        singleWordData["right"] = i.right;
        singleWordData["bottom"] = i.bottom;
        singleWordData["width"] = i.width;
        singleWordData["height"] = i.height;
        singleWordData["prev"] = i.distancePrevX;
        singleWordData["next"] = i.distanceNextX;
        singleWordData["above"] = i.distanceAboveY;
        singleWordData["below"] = i.distanceBelowY;
        singleWordData["is_table"] = i.isTable;
        singleWordData["image_id"] = justName;

        outFileWordData[ii] = singleWordData;

        ii++;
    }

    ofstream outStream(fileNameFeatures);
    outStream << outFileWordData;


    return 0;
}