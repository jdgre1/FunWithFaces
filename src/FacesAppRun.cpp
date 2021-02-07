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
		cv::dnn::Net dnnNet = dnn::readNetFromCaffe(fl.getModelTxt(), fl.getModel());
		// Does not work with CUDA YET!!
		dnnNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		dnnNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
		fl.setDnnNet(dnnNet);
		cerr << "loaded successfully" << endl;

	}

	catch (cv::Exception& e)
	{
		std::cerr << "loading failed: " << e.what() << std::endl;
		return;
	}


	cv::Mat michael_jordan = cv::imread("../data/mj.jpg", IMREAD_COLOR);
	cv::Mat theresa_may = cv::imread("../data/theresa_may.jpg", IMREAD_COLOR);
	cv::Mat putin = cv::imread("../data/putin.jpg", IMREAD_COLOR);



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


