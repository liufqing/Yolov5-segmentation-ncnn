#pragma once

#include <opencv2/core/mat.hpp>
#include <vector>

typedef std::vector<cv::Point2f> vectorpair;

namespace imutils {
    /**
    * @param points a vector of coordinates
    * @return Consistent order of points
    * @brief Converting vector of points to a vector with consistent order.
    **/
    vectorpair orderPoints(vectorpair points);

    /**
    * @param img Image
    * @param points a vector of coordinates
    * @return cropped Image with FPT applied.
    * @brief Applying four point transformation to the image
    **/
    cv::Mat fourPointTransformation(cv::Mat& img, vectorpair points);
};