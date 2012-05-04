#include <iostream>

#include <irrlicht.h>
#include <Poco/Foundation.h>

#include <opencv2/opencv.hpp>

#include "brisk.h"

#include "Utils.h"
#include "AR.h"

#if defined WIN32
#if defined _DEBUG
#pragma comment(lib,"opencv_core231d.lib")
#pragma comment(lib,"opencv_imgproc231d.lib")
#pragma comment(lib,"opencv_highgui231d.lib")
#pragma comment(lib, "opencv_features2d231d.lib")
#pragma comment(lib, "opencv_flann231d.lib")
#pragma comment(lib, "opencv_calib3d231d.lib")
#pragma comment(lib, "opencv_video231d.lib")
#pragma comment(lib, "PocoFoundation.lib")
#pragma comment(lib, "PocoXML.lib")
#pragma comment(lib, "Irrlicht.lib")
#else
#pragma comment(lib,"opencv_core231.lib")
#pragma comment(lib,"opencv_imgproc231.lib")
#pragma comment(lib,"opencv_highgui231.lib")
#pragma comment(lib, "opencv_features2d231.lib")
#pragma comment(lib, "opencv_flann231.lib")
#pragma comment(lib, "opencv_calib3d231.lib")
#pragma comment(lib, "opencv_video231.lib")
#pragma comment(lib, "PocoFoundationd.lib")
#pragma comment(lib, "PocoXMLd.lib")
#pragma comment(lib, "Irrlicht.lib")
#endif

#pragma warning(disable: 4251)
#pragma warning(disable: 4996)
#endif

using namespace std;

using namespace irr;
using namespace core;
using namespace scene;
using namespace video;
using namespace io;
using namespace gui;

//#pragma comment(linker, "/subsystem:windows /ENTRY:mainCRTStartup")

using namespace Poco;

using namespace cv;

using namespace ar;

const int WIN_WIDTH  = 640;
const int WIN_HEIGHT = 480;

#define MIN_TRACK_PT_NUM 8
#define MAX_TRACK_PT_NUM 200

const char *tmplImgPath = "../media/lena.png";

bool gRunning = true;

vector<Point2f> trackPtsPre;
vector<Point2f> trackPtsCur;

bool KeyPointReponseLarger(const KeyPoint &first, const KeyPoint &second)
{
	return first.response > second.response;
}

bool initTrackingPoints(Mat &queryImg)
{
	Mat H;
	Mat templImg = imread(tmplImgPath);
	if(templImg.empty()) return false;

	vector<KeyPoint> templKeys;
	Mat templDesc;

	vector<KeyPoint> queryKeys;
	Mat queryDesc;


	cv::Ptr<cv::FeatureDetector> detector;
	double keyThresh = 30;
	detector = new BriskFeatureDetector(keyThresh,4);

	cv::Ptr<cv::DescriptorExtractor> descriptorExtractor;
	descriptorExtractor = new BriskDescriptorExtractor();

	Mat templGray;
	cvtColor(templImg, templGray, CV_BGR2GRAY);

	detector->detect(templGray, templKeys);
	descriptorExtractor->compute(templGray, templKeys, templDesc);

	Mat queryGray;
	cvtColor(queryImg, queryGray, CV_BGR2GRAY);

	detector->detect(queryGray, queryKeys);
	descriptorExtractor->compute(queryGray, queryKeys, queryDesc);

	// find matches
	vector<DMatch> _matches;

	cv::Ptr<cv::DescriptorMatcher> descriptorMatcher;
	descriptorMatcher = new cv::BruteForceMatcher<cv::HammingSse>();
	descriptorMatcher->match(queryDesc, templDesc, _matches);

	vector<DMatch> matches;
	for(int i = 0; i < (int)_matches.size(); i++){
		if(_matches[i].distance < 100) matches.push_back(_matches[i]);
	}

	cout << "match keypoint number: " << matches.size() << endl;

	// convert keys to center aligned.
	float scale = 1;
	for(int i = 0; i < (int) templKeys.size(); i++){
		/*templKeys[i].pt.x = (templKeys[i].pt.x-templImg.cols/2)/scale;
		templKeys[i].pt.y = (templImg.rows/2 - templKeys[i].pt.y)/scale;*/
		templKeys[i].pt.x = (templKeys[i].pt.x)/scale;
		templKeys[i].pt.y = (templKeys[i].pt.y)/scale;
	}
	// find homography
	// queryKeys = H*TemplKeys
	vector<DMatch> goodMatches;
	findHomography(queryKeys, templKeys, matches, goodMatches, H, CV_RANSAC, 3);

	cout << "good match keypoint number: " << goodMatches.size() << endl;

	// find good point to track.
	vector<KeyPoint> trackPtCandis;
	for( int i = 0; i < (int) goodMatches.size(); i++ ){
		trackPtCandis.push_back( queryKeys[goodMatches[i].queryIdx] );
	}

	sort(trackPtCandis.begin(), trackPtCandis.end(), KeyPointReponseLarger);

	if( trackPtCandis.size() < MIN_TRACK_PT_NUM ) return false;
	// pick the first 200
	trackPtsPre.clear();
	trackPtsCur.clear();
	for( int i = 0; i < MIN(MAX_TRACK_PT_NUM, trackPtCandis.size()); i++ ){
		trackPtsPre.push_back(trackPtCandis[i].pt);
	}

	return true;

}

void help()
{
	// print a welcome message, and the OpenCV version
	cout << "\nThis is a demo of Lukas-Kanade optical flow lkdemo(),\n"
		"Using OpenCV version %s\n" << CV_VERSION << "\n"
		<< endl;

	cout << "\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - auto-initialize tracking\n"
		"\tc - delete all the points\n"
		"\tn - switch the \"night\" mode on/off\n"
		"To add/remove a feature point click it\n" << endl;
}

Point2f pt;
bool addRemovePt = false;

void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
	if( event == CV_EVENT_LBUTTONDOWN )
	{
		pt = Point2f((float)x,(float)y);
		addRemovePt = true;
	}
}

int main( int argc, char** argv )
{
	VideoCapture cap;
	TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);
	Size subPixWinSize(10,10), winSize(31,31);

	const int MAX_COUNT = 500;
	bool needToInit = false;
	bool nightMode = false;

	cap.open(0);

	if( !cap.isOpened() )
	{
		cout << "Could not initialize capturing...\n";
		return 0;
	}

	help();

	namedWindow( "PlaneTracking", 1 );
	setMouseCallback( "PlaneTracking", onMouse, 0 );

	Mat gray, prevGray, image;

	for(;;)
	{
		Mat frame;
		cap >> frame;
		if( frame.empty() )
			break;

		frame.copyTo(image);
		cvtColor(image, gray, CV_BGR2GRAY); 

		if( nightMode )
			image = Scalar::all(0);

		if( needToInit )
		{
			// automatic initialization
			/*goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termcrit);*/

			initTrackingPoints(frame);
			addRemovePt = false;
		}
		else if( !trackPtsPre.empty() )
		{
			vector<uchar> status;
			vector<float> err;
			if(prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, trackPtsPre, trackPtsCur, status, err, winSize,
				3, termcrit, 0, 0, 0.001);

			size_t i, k;
			for( i = k = 0; i < trackPtsCur.size(); i++ )
			{
				if( addRemovePt )
				{
					if( norm(pt - trackPtsCur[i]) <= 5 )
					{
						addRemovePt = false;
						continue;
					}
				}

				if( !status[i] )
					continue;

				trackPtsCur[k++] = trackPtsCur[i];
				circle( image, trackPtsCur[i], 3, Scalar(0,255,0), -1, 8);
			}
			trackPtsCur.resize(k);
		}

		if( addRemovePt && trackPtsCur.size() < (size_t)MAX_COUNT )
		{
			vector<Point2f> tmp;
			tmp.push_back(pt);
			cornerSubPix( gray, tmp, winSize, cvSize(-1,-1), termcrit);
			trackPtsCur.push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		imshow("LK Demo", image);

		char c = (char)waitKey(10);
		if( c == 27 )
			break;
		switch( c )
		{
		case 'r':
			needToInit = true;
			break;
		case 'c':
			trackPtsCur.clear();
			break;
		case 'n':
			nightMode = !nightMode;
			break;
		default:
			;
		}

		std::swap(trackPtsCur, trackPtsPre);
		swap(prevGray, gray);
	}

	return 0;
}