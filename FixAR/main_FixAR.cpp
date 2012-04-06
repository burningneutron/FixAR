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

Mat getHomography(Mat &queryImg)
{
	Mat H;
	Mat templImg = imread("../media/cvchina.jpg");
	if(templImg.empty()) return H;

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

	// find homography
	// queryKeys = H*TemplKeys
	vector<DMatch> goodMatches;
	findHomography(queryKeys, templKeys, matches, goodMatches, H, CV_RANSAC, 3);

	cout << "good match keypoint number: " << goodMatches.size() << endl;

	if(H.empty()) return H;
	else{
		cout << H << endl;
		H = H.inv(DECOMP_SVD);
		cout << H << endl;

		//normalize
		H = H * (1. / H.at<double>(2,2));
	}

	return H;
}
int main(int argc, char *argv[])
{

	IrrlichtDevice *devices = createDevice( video::EDT_OPENGL, dimension2d<u32>(WIN_WIDTH, WIN_HEIGHT), 16, false, false, false, 0);

	if(!devices) return 1;

	devices->setWindowCaption(L"FixAR demo");

	IVideoDriver *driver = devices->getVideoDriver();
	ISceneManager *smgr = devices->getSceneManager();
	IGUIEnvironment *guienv = devices->getGUIEnvironment();

	guienv->addStaticText(L"This is a FixAR demo.", rect<s32>(10, 10, 260, 22), true);

	IAnimatedMesh *mesh = smgr->getMesh("../media/sydney.md2");
	if(!mesh){
		devices->drop();
		return 1;
	}
	IAnimatedMeshSceneNode *node = smgr->addAnimatedMeshSceneNode(mesh);

	if( node ){
		node->setMaterialFlag(EMF_LIGHTING, false);
		node->setMD2Animation(scene::EMAT_STAND);
		node->setMaterialTexture(0, driver->getTexture("../media/sydney.bmp"));
	}

	// Add world 3-axis
	scene::ISceneNode *w_y_axis = smgr->addMeshSceneNode(
		smgr->addArrowMesh("w_y-axis",
		video::SColor(255, 0, 200, 0),
		video::SColor(255, 0, 255, 0),
		4, 8, 10.f, 9.f, 0.2f, 0.5f)
		);

	{

		scene::ISceneNode *w_x_axis = smgr->addMeshSceneNode(
			smgr->addArrowMesh("w_x-axis",
			video::SColor(255, 200, 0, 0),
			video::SColor(255, 255, 0, 0),
			4, 8, 10.f, 9.f, 0.2f, 0.5f)
			);

		scene::ISceneNode *w_z_axis = smgr->addMeshSceneNode(
			smgr->addArrowMesh("w_z-axis",
			video::SColor(255, 0, 0, 200),
			video::SColor(255, 0, 0, 255),
			4, 8, 10.f, 9.f, 0.2f, 0.5f)
			);

		w_y_axis->addChild(w_x_axis);
		w_y_axis->addChild(w_z_axis);

		
		w_x_axis->setMaterialFlag(video::EMF_LIGHTING, false);
		w_x_axis->setPosition(core::vector3df(0, 0, 0));
		w_x_axis->setRotation(core::vector3df(0, 0, -90));

		w_z_axis->setMaterialFlag(video::EMF_LIGHTING, false);
		w_z_axis->setPosition(core::vector3df(0, 0, 0));
		w_z_axis->setRotation(core::vector3df(90, 0, 0));

		w_y_axis->setMaterialFlag(video::EMF_LIGHTING, false);

		w_y_axis->setVisible(true);
	}

	// Add object 3-axis
	scene::ISceneNode *m_y_axis = smgr->addMeshSceneNode(
		smgr->addArrowMesh("m_y-axis",
		video::SColor(255, 0, 200, 0),
		video::SColor(255, 0, 255, 0),
		8, 18, 10.f, 5.f, 0.2f, 0.5f)
		);

	{

		scene::ISceneNode *m_x_axis = smgr->addMeshSceneNode(
			smgr->addArrowMesh("m_x-axis",
			video::SColor(255, 200, 0, 0),
			video::SColor(255, 255, 0, 0),
			8, 18, 10.f, 5.f, 0.2f, 0.5f)
			);

		scene::ISceneNode *m_z_axis = smgr->addMeshSceneNode(
			smgr->addArrowMesh("m_z-axis",
			video::SColor(255, 0, 0, 200),
			video::SColor(255, 0, 0, 255),
			8, 18, 10.f, 5.f, 0.2f, 0.5f)
			);

		m_y_axis->addChild(m_x_axis);
		m_y_axis->addChild(m_z_axis);


		m_x_axis->setMaterialFlag(video::EMF_LIGHTING, false);
		m_x_axis->setPosition(core::vector3df(0, 0, 0));
		m_x_axis->setRotation(core::vector3df(0, 0, -90));

		m_z_axis->setMaterialFlag(video::EMF_LIGHTING, false);
		m_z_axis->setPosition(core::vector3df(0, 0, 0));
		m_z_axis->setRotation(core::vector3df(90, 0, 0));

		m_y_axis->setMaterialFlag(video::EMF_LIGHTING, false);

		m_y_axis->setVisible(true);
	}


	ITexture *videoTexture = driver->addTexture(vector2d<u32>(WIN_WIDTH, WIN_HEIGHT), "video_stream");

	smgr->addCameraSceneNode(0, vector3df(0, 30, -40), vector3df(0, 5, 0));


	// open camera.
	VideoCapture capture(0);
	if( !capture.isOpened() ) return 1;
	Mat videoFrame;
	{
		int i = 0;
		while( i++ < 10 ){
			capture.grab();
			capture.retrieve(videoFrame, 0);
		}
	}

	// calc homography between the captured image and the template image.
	Mat H = getHomography(videoFrame);
	if(H.empty()){
		cout << "Homography estimation failed." << endl;
		return 1;
	}

	// load camera intrices matrix
	Mat K = loadCameraIntrinsic("../media/intrinsic.txt");

	// get model matrix.
	Pose pose;
	calcRTFromHomography(K, H, pose);

	cout << pose.translation_ << endl;

	while(devices->run()){
		driver->beginScene(true, true, SColor(255, 100, 101, 140));

		// draw video background Image
		// grab video frame.
		capture.grab();
		capture.retrieve(videoFrame);
		unsigned char *video_buf = (unsigned char*)videoTexture->lock();
		unsigned char *frame_buf = videoFrame.data;

		// Convert from RGB to RGBA
		for(int y=0; y < videoFrame.rows; y++) {
			for(int x=0; x < videoFrame.cols; x++) {
				*(video_buf++) = *(frame_buf++);
				*(video_buf++) = *(frame_buf++);
				*(video_buf++) = *(frame_buf++);
				*(video_buf++) = 128;
			}
		}
		videoTexture->unlock();

		// draw video background
		driver->draw2DImage(videoTexture, core::rect<s32>(0,0,WIN_WIDTH,WIN_HEIGHT), core::rect<s32>(0,0,WIN_WIDTH,WIN_HEIGHT));

		// set ar scene.
		float yaw, pitch, roll;
		getYPR(pose.rotation_, yaw, pitch, roll);
		vector3df rotation(yaw, pitch, roll);
		vector3df position;
		position.X = (float) (pose.translation_.at<double>(0, 0));
		position.Y = (float) (pose.translation_.at<double>(1, 0));
		position.Z = (float) (pose.translation_.at<double>(2, 0));

		//node->setRotation(rotation);
		node->setPosition(position);
		node->setVisible(false);

		//m_y_axis->setRotation(rotation);
		m_y_axis->setPosition(position);

		smgr->drawAll();
		guienv->drawAll();
		driver->endScene();
	}

	devices->drop();

	return 0;
}