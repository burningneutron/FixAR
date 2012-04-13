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

bool gRunning = true;
Pose gPose;

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

	if(H.empty()) return H;
	else{
		cout << H << endl;
		H = H.inv(DECOMP_SVD);

		//normalize
		H = H * (1. / H.at<double>(2,2));
		cout << H << endl;


		/*Mat showImg;
		warpPerspective (templImg, showImg, H, Size(640,480));
		imshow("wrap", showImg);
		cvWaitKey(0);*/
	}

	vector<Point2f> npts;
	vector<Point2f> tpts;
	npts.resize(matches.size());
	tpts.resize(matches.size());
	for(int i = 0; i < (int)matches.size(); i++){
		npts[i] = queryKeys[matches[i].queryIdx].pt;
		tpts[i] = templKeys[matches[i].trainIdx].pt;
	}
	H = findHomography(Mat(tpts), Mat(npts), CV_RANSAC, 3);

	cout << " alt H: " << endl;
	cout << H << endl;

	return H;
}

void updatePose()
{

}

class UpdaePoseEventReceiver: public IEventReceiver
{
public:
	virtual bool onEvent(const SEvent& event)
	{
		if(event.EventType == irr::EET_KEY_INPUT_EVENT){
			if(event.KeyInput.Key == KEY_ESCAPE){
				gRunning = false;
			}

			if(event.KeyInput.Key == KEY_KEY_U){
				updatePose();
			}
		}
	}
};

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
	float arrowScale = 20.f;
	float arrowHeight = 10.f*arrowScale;
	float arrowCylinderHeight = 5.f*arrowScale;
	float arrowWidth0 = 0.2f*arrowScale;
	float arrowWidth1 = 0.5f*arrowScale;
	scene::ISceneNode *m_y_axis = smgr->addMeshSceneNode(
		smgr->addArrowMesh("m_y-axis",
		video::SColor(255, 0, 200, 0),
		video::SColor(255, 0, 255, 0),
		8, 18, arrowHeight, arrowCylinderHeight, arrowWidth0, arrowWidth1)
		);

	{

		scene::ISceneNode *m_x_axis = smgr->addMeshSceneNode(
			smgr->addArrowMesh("m_x-axis",
			video::SColor(255, 200, 0, 0),
			video::SColor(255, 255, 0, 0),
			8, 18, arrowHeight, arrowCylinderHeight, arrowWidth0, arrowWidth1)
			);

		scene::ISceneNode *m_z_axis = smgr->addMeshSceneNode(
			smgr->addArrowMesh("m_z-axis",
			video::SColor(255, 0, 0, 200),
			video::SColor(255, 0, 0, 255),
			8, 18, arrowHeight, arrowCylinderHeight, arrowWidth0, arrowWidth1)
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

	ICameraSceneNode *cam =  smgr->addCameraSceneNode(0, vector3df(0, 0, 0), vector3df(0, 0, 1));
	cam->setNearValue(1);
	cam->setFarValue(3000);

	{

	matrix4 oldProj = cam->getProjectionMatrix();

	cout << " Proj: " << endl;
	cout << oldProj[0] << " "
		 << oldProj[1] << " "
		 << oldProj[2] << " "
		 << oldProj[3] << " " << endl
		 << oldProj[4] << " "
		 << oldProj[5] << " "
		 << oldProj[6] << " "
		 << oldProj[7] << " " << endl
		 << oldProj[8] << " "
		 << oldProj[9] << " "
		 << oldProj[10] << " "
		 << oldProj[11] << " " << endl
		 << oldProj[12] << " " 
		 << oldProj[13] << " "
		 << oldProj[14] << " "
		 << oldProj[15] << " " << endl;
	}

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

	Mat _H = Mat::eye(3,3,CV_64F);
	_H = _H * 0.5;
	Mat testFrame;
	Mat templImg = imread("../media/cvchina.jpg");
	warpPerspective(templImg, testFrame, _H, Size(640,480));

	// calc homography between the captured image and the template image.
	Mat H = getHomography(videoFrame);
	if(H.empty()){
		cout << "Homography estimation failed." << endl;
		return 1;
	}

	// load camera intrices matrix
	Mat K = loadCameraIntrinsic("../media/intrinsic.txt");

	//matrix4 proj = oldProj;
	//proj[0] = K.at<double>(0, 0);
	//proj[4] = K.at<double>(1, 1);
	//proj[2] = K.at<double>(0, 2);
	//proj[5] = K.at<double>(1, 2);

	//cam->setProjectionMatrix(proj);
	float fov = 2*atan2(320, K.at<double>(0, 0));
	cout << "fov: " << fov << endl;

	cam->setFOV((float) fov );
	cam->setAspectRatio(640/480.f);

	cout << "fov: " << cam->getFOV() << endl;

	// get model matrix.
	calcRTFromHomography(K, H, gPose);
	//calcRTUsingRPP(K, H, gPose);

	// set ar scene.
	float yaw, pitch, roll;
	getYPR(gPose.rotation_, yaw, pitch, roll);

	cout << "yaw: " << yaw << " pitch: " << pitch << " roll: " << roll << endl;

	Mat rotationVector;
	Rodrigues(gPose.rotation_, rotationVector);

	yaw = rotationVector.at<double>(0, 0) * 180 / CV_PI;
	pitch = rotationVector.at<double>(1, 0) * 180 / CV_PI;
	roll = rotationVector.at<double>(2, 0) * 180 / CV_PI;

	cout <<" rotation vector: " << rotationVector << endl;

	cout << "yaw: " << yaw << " pitch: " << pitch << " roll: " << roll << endl;

	cout << gPose.translation_ << endl;

	while(devices->run() && gRunning){
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

		vector3df rotation(-yaw, pitch, roll);
		vector3df position;
		position.X = (float) (gPose.translation_.at<double>(0, 0));
		position.Y = (float) (-gPose.translation_.at<double>(1, 0));
		position.Z = (float) (gPose.translation_.at<double>(2, 0));
	

		//position.X = -20; position.Y = 0; position.Z = 2000;

		node->setRotation(rotation);
		node->setPosition(position);
		node->setVisible(false);

		w_y_axis->setVisible(false);

		m_y_axis->setRotation(rotation);
		m_y_axis->setPosition(position);

		smgr->drawAll();
		guienv->drawAll();
		driver->endScene();
	}

	devices->drop();

	return 0;
}