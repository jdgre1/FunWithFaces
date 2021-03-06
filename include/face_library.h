#pragma once
#ifndef FACE_LIBRARY__H__
#define FACE_LIBRARY__H__

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

	// 3. Copy constructor  //FaceLibrary(const FaceLibrary&) = delete; // (if to disable copy constructor) 
	                     
	// DNN Model related methods
	std::string get_model_txt() const;
	std::string get_model() const;

	void set_dnn_net(const cv::dnn::Net&);

	// Facebox methods
	void return_facebox(cv::Mat&, cv::Rect&);
	cv::Rect find_best_face_position(cv::Mat&, cv::Mat&);
	void swap_faces(cv::Mat&, cv::Rect&, cv::Mat&, cv::Rect&);
	

private:
	std::string m_random_crap = "hello";
	FaceParams m_faceParams;
	std::string m_modelTxt = "../../models/deploy.prototxt.txt";      //"../../models/face_detector.prototxt.txt"; // "../../models/MobileNetSSD_deploy.prototxt.txt";  
	std::string m_modelBin = "../../models/res10_300x300_ssd_iter_140000.caffemodel";        //"../../models/face_detector.caffemodel"; // "../../models/MobileNetSSD_deploy.caffemodel";  
	cv::dnn::Net m_dnnNet;
	
	
};


#endif