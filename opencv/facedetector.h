#include <stdio.h>

#include "opencv2/opencv.hpp"

class FaceDetector
{
private:
    cv::CascadeClassifier faceCascade;
    cv::CascadeClassifier eyeCascade1;
    cv::CascadeClassifier eyeCascade2;
    void detectBothEyes(const cv::Mat &face, cv::Point &leftEye, cv::Point &rightEye);
    void equalizeLeftAndRightHalves(cv::Mat &faceImg);
    void detectObjects(const cv::Mat &img, cv::CascadeClassifier &cascade, std::vector<cv::Rect> &objects, int scaledWidth, int flags, cv::Size minFeatureSize, float searchScaleFactor, int minNeighbors);
    void findLargestObject(const cv::Mat &img, cv::CascadeClassifier &cascade, cv::Rect &largestObject, int scaledWidth = 320);
public:
    void initCascadeClassifiers();
    cv::Mat getPreprocessedFace(cv::Mat srcImg);
};