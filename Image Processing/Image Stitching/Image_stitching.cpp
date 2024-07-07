#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void stitchImages(const Mat& img1, const Mat& img2, Mat& output) {
    // Step 1: Detect and describe keypoints using SIFT
    Ptr<SIFT> sift = SIFT::create();
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    
    sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    
    // Step 2: Match features using BFMatcher
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    // Filter matches based on distance
    double maxDist = 0;
    double minDist = 100;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }
    
    vector<DMatch> goodMatches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance <= max(2 * minDist, 0.02)) {
            goodMatches.push_back(matches[i]);
        }
    }
    
    // Step 3: Find homography using RANSAC
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < goodMatches.size(); i++) {
        points1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
        points2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
    }
    
    Mat H = findHomography(points2, points1, RANSAC);
    
    // Step 4: Warp the second image to align with the first image
    Mat img2Warped;
    warpPerspective(img2, img2Warped, H, Size(img1.cols + img2.cols, img1.rows));
    
    // Copy the first image into the output image
    Mat half(output, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);
    
    // Blend the images
    addWeighted(half, 0.5, img2Warped(Rect(0, 0, img1.cols, img1.rows)), 0.5, 0, half);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: " << argv[0] << " <image1> <image2>" << endl;
        return -1;
    }
    
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
    
    if (img1.empty() || img2.empty()) {
        cout << "Error loading images!" << endl;
        return -1;
    }
    
    Mat output;
    stitchImages(img1, img2, output);
    
    imshow("Stitched Image", output);
    waitKey(0);
    
    return 0;
}