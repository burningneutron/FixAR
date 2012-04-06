#include "Utils.h"

using namespace std;
using namespace cv;

namespace ar
{
	void findHomography(
		const std::vector<cv::KeyPoint> &queryKeys,
		const std::vector<cv::KeyPoint> &templKeys,
		const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inlier, cv::Mat &H, int method, float ransacReprojThreshold)
	{
		int matchNum = (int)matches.size();
		Mat pt1(Size(2, matchNum), CV_32FC1);
		Mat pt2(Size(2, matchNum), CV_32FC1);

		if( matchNum < 4 ){
			H.release();
			inlier.clear();
			return;
		}

		for( int i = 0; i < matchNum; i++ ){
			pt1.at<float>(i, 0) = queryKeys[ matches[i].queryIdx ].pt.x;
			pt1.at<float>(i, 1) = queryKeys[ matches[i].queryIdx ].pt.y;

			pt2.at<float>(i, 0) = templKeys[ matches[i].trainIdx ].pt.x;
			pt2.at<float>(i, 1) = templKeys[ matches[i].trainIdx ].pt.y;
		}

		H = findHomography(pt1, pt2, method, ransacReprojThreshold);
		if( H.empty() ){
			inlier.clear();
			return;
		}

		float thresh = ransacReprojThreshold * ransacReprojThreshold;

		for( int i = 0; i < matchNum; i++ ){
			Point2f _pt1 = queryKeys[ matches[i].queryIdx ].pt;
			Point2f _pt2 = templKeys[ matches[i].trainIdx ].pt;
			Point2f _mappedPt;
			float denom;
			denom = 1.f / (float) ( H.at<double>(2, 0) * _pt1.x + H.at<double>(2, 1) * _pt1.y + H.at<double>(2, 2) );

			_mappedPt.x = (float) ( H.at<double>(0, 0) * _pt1.x + H.at<double>(0, 1) * _pt1.y + H.at<double>(0, 2) ) * denom;
			_mappedPt.y = (float) ( H.at<double>(1, 0) * _pt1.x + H.at<double>(1, 1) * _pt1.y + H.at<double>(1, 2) ) * denom;

			float err;
			err = (_mappedPt.x - _pt2.x)*(_mappedPt.x - _pt2.x) + (_mappedPt.y - _pt2.y)*(_mappedPt.y - _pt2.y);
			if( err < thresh ) inlier.push_back(matches[i]);
		}
	}
}
