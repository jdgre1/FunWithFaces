#include <pch.h> 
#include <face_library.h>
#include <filesystem>

using namespace cv;
using namespace std;

namespace fs = std::filesystem;


bool getDesktopResolution(int& screenHeight, int& screenWidth);


void RunFaceApp() {

	FaceLibrary fl;

	try {
		cv::dnn::Net dnn_net = dnn::readNetFromCaffe(fl.get_model_txt(), fl.get_model());
		dnn_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		dnn_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		fl.set_dnn_net(dnn_net);
		cerr << "loaded successfully" << endl;

	}

	catch (cv::Exception& e)
	{
		std::cerr << "loading failed: " << e.what() << std::endl;
		return;
	}

	cv::Mat michael_jordan = cv::imread("../../data/mj.jpg", IMREAD_COLOR);
	cv::Mat theresa_may = cv::imread("../../data/theresa_may.jpg", IMREAD_COLOR);
	cv::Mat putin = cv::imread("../../data/putin.jpg", IMREAD_COLOR);
	cv::Rect mj_facebox, tm_facebox;
	fl.return_facebox(michael_jordan, mj_facebox);
	fl.return_facebox(theresa_may, tm_facebox);
	
	cout << mj_facebox.x << " , " << mj_facebox.y << " , " << mj_facebox.width << " , "  << mj_facebox.height;
	fl.swap_faces(michael_jordan, mj_facebox, theresa_may, tm_facebox);
}


bool getDesktopResolution(int& screenWidth, int& screenHeight) {

	RECT desktop;
	// Get a handle to the desktop window
	const HWND hDesktop = GetDesktopWindow();
	// Get the size of screen to the variable desktop
	GetWindowRect(hDesktop, &desktop);
	// The top left corner will have coordinates (0,0)
	// and the bottom right corner will have coordinates
	// (horizontal, vertical)
	screenWidth = desktop.right;
	screenHeight = desktop.bottom;

	return ((desktop.right > 0) & (desktop.bottom > 0)) ? true : false;
}


