#ifndef  __AR_H__
#define  __AR_H__

#include <opencv2/opencv.hpp>
namespace ar
{
	class Pose
	{
	public:
		cv::Mat translation_;
		cv::Mat rotation_;
		cv::Mat scale_;
		float yaw_, pitch_, roll_;
	};

	bool calcRTFromHomography(const cv::Mat &K, const cv::Mat &H, Pose &pose);
	void getYPR(const cv::Mat &rotation, float &yaw, float &pitch, float &roll);

	cv::Mat loadCameraIntrinsic(std::string path);
}
#endif