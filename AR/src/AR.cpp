#include "AR.h"
#include "RPP.h"
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
		Mat _H;
#ifdef DEBUG_POSE_ESTIMATE
		cout << "H: " << endl;
		cout << H << endl;
#endif
		if(determinant(H) < 0){
#ifdef DEBUG_POSE_ESTIMATE
			cout << "det(H) is < 0 " << endl;
#endif
			_H = H * -1;
		}else{
			H.copyTo(_H);
		}

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
		A = invK*_H;

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

		// refine R
		/*Mat U, VT;
		SVD svd(R);

		R = svd.u.mul(svd.vt);*/

#ifdef DEBUG_POSE_ESTIMATE
		cout <<"R'*R: " << endl;
		cout << R.t() * R << endl;

		cout << "det(R) = " << determinant(R) << endl;
#endif

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

	bool calcRTUsingRPP(const cv::Mat &K, const cv::Mat &H,Pose &pose)
	{
		Mat invK = K.inv(DECOMP_SVD);
		Mat imgPts = Mat::zeros(3, 4, CV_64F);

		const double objWidth = 512;
		const double objHeight = 496;

		// 4 corner points of the model
		if(1){
			// top left
			imgPts.at<double>(0,0) = 0;
			imgPts.at<double>(1,0) = 0;
			imgPts.at<double>(2,0) = 1;

			// top right
			imgPts.at<double>(0,1) = objWidth;
			imgPts.at<double>(1,1) = 0;
			imgPts.at<double>(2,1) = 1;

			// bottom left
			imgPts.at<double>(0,2) = objWidth;
			imgPts.at<double>(1,2) = objHeight;
			imgPts.at<double>(2,2) = 1;

			// bottom right
			imgPts.at<double>(0,3) = 0;
			imgPts.at<double>(1,3) = objHeight;
			imgPts.at<double>(2,3) = 1;
		}else{
			// top left
			imgPts.at<double>(0,0) = 0;
			imgPts.at<double>(1,0) = objHeight;
			imgPts.at<double>(2,0) = 1;

			// top right
			imgPts.at<double>(0,1) = objWidth;
			imgPts.at<double>(1,1) = objHeight;
			imgPts.at<double>(2,1) = 1;

			// bottom left
			imgPts.at<double>(0,2) = 0;
			imgPts.at<double>(1,2) = 0;
			imgPts.at<double>(2,2) = 1;

			// bottom right
			imgPts.at<double>(0,3) = objWidth;
			imgPts.at<double>(1,3) = 0;
			imgPts.at<double>(2,3) = 1;
		}

		double scale = 1.0 / (double)objWidth;
		cv::Mat normMat = cv::Mat::eye(3,3,CV_64F);
		normMat.at<double>(0,0) = scale;
		normMat.at<double>(1,1) = scale;
		normMat.at<double>(0,2) = -scale*objWidth*0.5;
		normMat.at<double>(1,2) = -scale*objHeight*0.5;

		Mat objPts = normMat*imgPts;
		for(int i=0; i < objPts.cols; i++) {
			objPts.at<double>(2,i) = 0.0; // z value
		}

		imgPts = invK * H * imgPts;
		
		double objErr, imgErr;
		int it;

		Mat R, T;
		bool status = Rpp(objPts, imgPts, R, T, it, objErr, imgErr);

#ifdef DEBUG_POSE_ESTIMATE
		cout << "rotation: " << endl;
		cout << R << endl;
		cout << "translation: " << endl;
		cout << T << endl;

		cout << "objErr: " << objErr << " imgErr: " << imgErr << endl;
#endif

		pose.rotation_ = R;
		pose.translation_ = T;
		
		return status;
	}

	void getYPR(const cv::Mat &rotation, float &yaw, float &pitch, float &roll)
	{
		// http://planning.cs.uiuc.edu/node103.html
		yaw = (float)( atan2(rotation.at<double>(1,0), rotation.at<double>(0,0)) * 180 / CV_PI );
		pitch = (float)( atan2(-rotation.at<double>(2,0),sqrt(rotation.at<double>(2,1)*rotation.at<double>(2,1) + rotation.at<double>(2,2)*rotation.at<double>(2,2)))  * 180 / CV_PI );
		roll = (float)( atan2(rotation.at<double>(2,1), rotation.at<double>(2,2))  * 180 / CV_PI );
	}

	cv::Mat loadCameraIntrinsic(std::string path)
	{
		Mat K = Mat::eye(3,3, CV_64F);
/*
		K.at<double>(0, 0) = 7.8619781114297075e+002;
		K.at<double>(1, 1) = 7.8665031379117136e+002;

		K.at<double>(0, 2) = 3.2587477764306675e+002;
		K.at<double>(1, 2) = 2.7700627412978264e+002;*/

		K.at<double>(0, 0) = 600;
		K.at<double>(1, 1) = 600;

		K.at<double>(0, 2) = 324;
		K.at<double>(1, 2) = 240;

		return K;
	}
}