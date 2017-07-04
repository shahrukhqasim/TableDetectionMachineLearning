//
// Created by srq on 7/4/17.
//

#include "helpers.h"

using namespace std;
using namespace cv;

bool horizontalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.y + i.height, j.y + j.height) - std::max(i.y, j.y)) > 0;
}

bool verticalOverlap(Rect i, Rect j) {
    return max(0, std::min(i.x + i.width, j.x + j.width) - std::max(i.x, j.x)) > 0;
}