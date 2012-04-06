#include "AR.h"
#include <iostream>

using namespace cv;
using namespace std;

namespace ar
{
#define DEBUG_POSE_ESTIMATE

	//H = s^(-1) * K * [r1,r2,T]
	bool calcRTFromHomography(const cv::Mat &K, const cv::Mat &H, Pose &pose)
	{
		// TODO:: flip if |H| < 0;
#ifdef DEBUG_POSE_ESTIMATE
		cout << "H: " << endl;
		cout << H << endl;
#endif
		assert(determinant(H) > 0);

#ifdef DEBUG_POSE_ESTIMATE
		cout << "K: " << endl;
		cout << K << endl;
#endif
		Mat invK = K.inv(DECOMP_SVD);


#ifdef DEBUG_POSE_ESTIMATE
		cout << " inv K: " << endl;
		cout << invK << endl;
#endif

		Mat A;
		A = invK*H;

#ifdef DEBUG_POSE_ESTIMATE
		cout <<"A: " << endl;
		cout << A << endl;
#endif

		double * _A = (double*) A.data;

		double s1 = sqrt(_A[0]*_A[0]+_A[3]*_A[3]+_A[6]*_A[6]);
		double s2 = sqrt(_A[1]*_A[1]+_A[4]*_A[4]+_A[7]*_A[7]);
		double s = 1.0/sqrt(s1*s2);

#ifdef DEBUG_POSE_ESTIMATE
		cout << "s: " << endl;
		cout << s << endl;
#endif
		A = s*A;

#ifdef DEBUG_POSE_ESTIMATE
		cout <<"s*A: " << endl;
		cout << A << endl;
#endif
		Mat T;
		A.col(2).copyTo(T);

		Mat R = A;
		Mat r1 = R.col(0);
		Mat r2 = R.col(1);
		Mat r3 = R.col(2);



		r1.cross(r2).copyTo(r3);

		// do polar decompostion  so R is orthogonal
		// R = (UV')(VSV')
		Mat U, VT;
		SVD svd(R);

		R = svd.u.mul(svd.vt);

		pose.rotation_ = R;
		pose.translation_ = T;
#ifdef DEBUG_POSE_ESTIMATE
		cout << "rotation: " << endl;
		cout << R << endl;
		cout << "translation: " << endl;
		cout << T << endl;
#endif

		return true;
	}

	void getYPR(const cv::Mat &rotation, float &yaw, float &pitch, float &roll)
	{
		// http://planning.cs.uiuc.edu/node103.html
		yaw = atan2(rotation.at<double>(1,0), rotation.at<double>(0,0));
		pitch = atan2(-rotation.at<double>(2,0),sqrt(rotation.at<double>(2,1)*rotation.at<double>(2,1) + rotation.at<double>(2,2)*rotation.at<double>(2,2)));
		roll = atan2(rotation.at<double>(2,1), rotation.at<double>(2,2));
	}

	cv::Mat loadCameraIntrinsic(std::string path)
	{
		Mat K = Mat::eye(3,3, CV_64F);

		K.at<double>(0, 0) = 1;
		K.at<double>(1, 1) = 1;

		K.at<double>(0, 2) = 320;
		K.at<double>(1, 2) = 240;

		return K;
	}
}