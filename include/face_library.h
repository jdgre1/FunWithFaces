#pragma once
#ifndef PAGE_PROCESSOR__H__
#define PAGE_PROCESSOR__H__

#include <pch.h>

#define HAS_CUDA 1

class FaceLibrary {

public:
	struct FaceParams {

		
		
	};


	// Constructor
	FaceLibrary();

	// Rule of 3
	// 1. Destructor
	~FaceLibrary();
	// 2. Copy assignment operator
	FaceLibrary& operator=(FaceLibrary&) = default;
	// 3. Copy constructor
	FaceLibrary(const FaceLibrary&);        //FaceLibrary(const FaceLibrary&) = delete; // (if to disable copy constructor) 
	
	std::string getModelTxt() const;
	std::string getModel() const;

	cv::Rect return_facebox(cv::Mat&);

	void setDnnNet(const cv::dnn::Net& net);

private:
	FaceParams m_faceParams;
	std::string m_modelTxt = "../../models/deploy.prototxt.txt";      //"../../models/face_detector.prototxt.txt"; // "../../models/MobileNetSSD_deploy.prototxt.txt";  
	std::string m_modelBin = "../../models/res10_300x300_ssd_iter_140000.caffemodel";        //"../../models/face_detector.caffemodel"; // "../../models/MobileNetSSD_deploy.caffemodel";  
	cv::dnn::Net m_dnnNet;
	
	
};


#endif