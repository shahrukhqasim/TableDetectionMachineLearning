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

using namespace std;
using namespace cv;


bool horizontalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.y + i.height, j.y + j.height) - std::max(i.y, j.y)) > 0;
}

bool verticalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.x + i.width, j.x + j.width) - std::max(i.x, j.x)) > 0;
}

struct Feature {
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
};

double scale(int max, int value) {
    return value;//min(1.00,((double)value)/max);
}

vector<Feature> findFeatures(std::vector<std::string> words, std::vector<Rect> wordBoxes, int imageWidth, int imageHeight) {
    vector<Feature> wordFeatures;

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

        Feature featureVector;
        featureVector.left = scale(imageWidth, i.x);
        featureVector.top = scale(imageHeight, i.y);
        featureVector.right = scale(imageWidth, i.x + i.width);
        featureVector.bottom = scale(imageHeight, i.y + i.height);
        featureVector.width = scale(imageWidth, i.width);
        featureVector.height = scale(imageHeight, i.height);
        featureVector.distancePrevX = leftFound ? scale(imageWidth, i.x - left.x - left.width) : -1;
        featureVector.distanceNextX = rightFound ? scale(imageWidth, right.x - i.x - i.width) : -1;
        featureVector.distanceBelowY = lowerFound ? scale(imageHeight, lower.y - i.y - i.height) : -1;
        featureVector.distanceAboveY = upperFound ? scale(imageHeight, i.y - upper.y - upper.height) : -1;

        wordFeatures.push_back(featureVector);
    }

    return wordFeatures;
}

int main(int argc, char**argv) {
    // Read the image
    if(argc!=2) {
        cout<<"Improper arguments \n";
        return -1;
    }

    string imageFileName = argv[1];
    Mat image;
    image = imread(imageFileName, 0);
    Mat imageRgb = imread(imageFileName, 1);

    // Binarize the image
    Mat binarizedImage;
    Mat temp;
    docproc::binarize::binarizeBG(image, temp, binarizedImage);

    //Tesseract Initialization
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    if (api->Init(NULL, "eng", tesseract::OEM_TESSERACT_ONLY)) {
        cerr << "Error in API initialization ";
        exit(-1);
    }

    //Set image to tesseract object
    api->SetImage((uchar *) binarizedImage.data, binarizedImage.size().width, binarizedImage.size().height,
                  binarizedImage.channels(), binarizedImage.step1());
    api->Recognize(0);
    tesseract::ResultIterator *ri = api->GetIterator();

    std::vector<std::string> words;
    std::vector<Rect> bboxes;

    // Get bounding boxes for each word in image
    if (ri != 0) {
        do {
            Rect bbox_Coordinates;
            string word =  ri->GetUTF8Text(tesseract::RIL_WORD);
            int left, top, right, bottom;
            ri->BoundingBox(tesseract::RIL_WORD, &left, &top, &right, &bottom);
            words.push_back(word);
            bboxes.push_back(Rect(left,top,right-left,bottom-top));

        } while (ri->Next(tesseract::RIL_WORD));
    }

    Mat image2;
    cvtColor(image, image2, CV_GRAY2BGR);

    for (auto i : bboxes) {
        rectangle(image2, i, Scalar(0,0,255), 3, 8, 0);
    }


    vector<Feature> featuresFile = findFeatures(words, bboxes, image.cols, image.rows);


    string withoutExtension = docproc::utility::getFileNameWithoutExtension(imageFileName);
    string fileNameFeatures = withoutExtension + ".csv";

    imwrite(withoutExtension+"_rects.png", image2);
    ofstream fileOutFeatures (fileNameFeatures);
    fileOutFeatures << "word,left,top,right,bottom,width,height,previous,next,above,below"<<endl;

    int ii = 0;
    for (auto i : featuresFile) {
        fileOutFeatures << words[ii] << ',' << i.left << ',' << i.top << ',' << i.right<< ',' << i.bottom<< ',';
        fileOutFeatures << i.width << ',' << i.height<< ','<< i.distancePrevX << ','<< i.distanceNextX<< ',';
        fileOutFeatures << i.distanceAboveY<< ','<< i.distanceBelowY << endl;
        ii++;
    }


    return 0;
}