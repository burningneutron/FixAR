#ifndef __UTILS_H__
#define __UTILS_H__
#include "opencv2/opencv.hpp"

#include <vector>
#include <string>

namespace ar
{

#define AR_SQUARE(x) ((x)*(x))
#define AR_MAX(x, y) ((x) > (y) ? (x) : (y))
#define AR_MIN(x, y) ((x) < (y) ? (x) : (y))

	// mapping queryKeys to templKeys
	void findHomography(
		const std::vector<cv::KeyPoint> &queryKeys,
		const std::vector<cv::KeyPoint> &templKeys,
		const std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &inliner, cv::Mat &H, int method = 0, float ransacReprojThreshold = 3.f);
}

#endif
