#include <face_library.h> 
//#include <fstream>
using namespace cv;
using namespace std;


FaceLibrary::FaceLibrary() {

	/*
    m_api = new tesseract::TessBaseAPI();
	if (m_api->Init(NULL, "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}
	m_api->SetDebugVariable("debug_file", "/dev/null");
    */
}

FaceLibrary::~FaceLibrary() {}

//FaceLibrary::FaceLibrary(const FaceLibrary&){
//}

std::string FaceLibrary::get_model_txt() const {
	return m_modelTxt;
}
std::string FaceLibrary::get_model() const {
	return m_modelBin;
}


// Credit: https://bewagner.net/programming/2020/04/12/building-a-face-detector-with-opencv-in-cpp/
cv::Rect FaceLibrary::find_best_face_position(cv::Mat& detection_matrix, cv::Mat& input_img) {

    float max_confidence_threshold_ = 0.165;
    int max_area = 0.;
    cv::Rect face;
    std::cout << "detection_matrix.size()" << detection_matrix.size() << std::endl;
    for (int i = 0; i < detection_matrix.rows; i++) {
        float confidence = detection_matrix.at<float>(i, 2);
        std::cout << "confidence = " << confidence << std::endl;
        if (confidence > max_confidence_threshold_) {
           
            int x_left_bottom = static_cast<int>(
                detection_matrix.at<float>(i, 3) * input_img.cols);

            int y_left_bottom = static_cast<int>(
                detection_matrix.at<float>(i, 4) * input_img.rows);

            int x_right_top = static_cast<int>(
                detection_matrix.at<float>(i, 5) * input_img.cols);

            int y_right_top = static_cast<int>(
                detection_matrix.at<float>(i, 6) * input_img.rows);

            int face_area = (y_right_top - y_left_bottom) * (x_right_top - x_left_bottom);
            if (face_area > max_area) {

                if (x_left_bottom >= 0 && y_left_bottom <= input_img.cols && x_right_top <= input_img.cols && y_right_top >= 0) {

                    max_area = face_area;
                    face.x = x_left_bottom;
                    face.y = y_left_bottom;
                    face.width = x_right_top - x_left_bottom;
                    face.height = y_right_top - y_left_bottom;
                }
            }
        }
    }

    return face;
}

void FaceLibrary::return_facebox(cv::Mat& img, cv::Rect& facebox) {
    
    int img_height = img.rows;
    int img_width = img.cols;
    cv::Mat blob_input;

    // Set input blob for detections
    cv::resize(img, blob_input, cv::Size(300, 300));
    Mat inputBlob = dnn::blobFromImage(blob_input, 1.0, Size(300, 300), Scalar(104.0, 177.0, 123.0), false);
    std::cout << "blob_input.size()" << blob_input.size() << std::endl;
    m_dnnNet.setInput(inputBlob, "data");


    // forward pass of neural network
    //std::vector<cv::Mat> detections = m_dnnNet.forward();
    cv::Mat detections = m_dnnNet.forward();
    detections = m_dnnNet.forward("detection_out");
    Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
    facebox = find_best_face_position(detectionMat, img);
    rectangle(img, facebox, Scalar(0, 255, 0), 1);
    
    return;
}

void FaceLibrary::set_dnn_net(const cv::dnn::Net& net)
{
	//cout << net.empty() << endl;
	m_dnnNet = net;
}

void FaceLibrary::swap_faces(cv::Mat& img1, cv::Rect& rect1, cv::Mat& img2, cv::Rect& rect2) {
    
    // define offset for defining the ROI outside the given face-box
    int roi_offset = 30;

    // define region of interest for both faces:
    cv::Rect face1_roi;
    face1_roi.x = rect1.x - roi_offset;
    face1_roi.y = rect1.y - roi_offset;
    face1_roi.width = rect1.width + roi_offset;
    face1_roi.height = rect1.height + roi_offset;
    cv::Mat face1_cropped = img1(face1_roi);

    cv::Rect face2_roi;
    face2_roi.x = rect2.x - roi_offset;
    face2_roi.y = rect2.y - roi_offset;
    face2_roi.width = rect2.width + roi_offset;
    face2_roi.height = rect2.height + roi_offset;
    cv::Mat face2_cropped = img2(face2_roi);
    
    // Create masks of both cropped images:
    cv::Mat src1_mask = cv::Mat::zeros(face1_cropped.size(), face1_cropped.type());
    cv::Mat src2_mask = cv::Mat::zeros(face2_cropped.size(), face2_cropped.type());
    
    const int num_points = 4;
    vector<vector<cv::Point>> face_rect1, face_rect2;
    face_rect1.resize(1);
    face_rect1[0].resize(num_points);
    face_rect2.resize(1);
    face_rect2[0].resize(num_points);
    
    face_rect1[0][0] = cv::Point(roi_offset * 2, roi_offset * 2);
    face_rect1[0][1] = cv::Point(face1_roi.width * 2 + roi_offset * 2, roi_offset * 2);
    face_rect1[0][2] = cv::Point(face1_roi.width * 2, face1_roi.height * 2 + roi_offset * 2);
    face_rect1[0][3] = cv::Point(roi_offset * 2, face1_roi.height * 2 + roi_offset * 2);
   
    // Point face_rect2[1][num_points];
    face_rect2[0][0] = cv::Point(roi_offset, roi_offset);
    face_rect2[0][1] = cv::Point(face2_roi.width + roi_offset, roi_offset);
    face_rect2[0][2] = cv::Point(face2_roi.width, face2_roi.height + roi_offset);
    face_rect2[0][3] = cv::Point(roi_offset, face2_roi.height + roi_offset);
    
    int lineType = LINE_8;
    cv::fillPoly(src1_mask, face_rect1, Scalar(255, 255, 255), lineType);
    cv::fillPoly(src2_mask, face_rect2, Scalar(255, 255, 255), lineType);

    // resize crop and mask images to double original size:
    cv::Mat src_mask1_resized, img1_resized, img2_resized, src_mask2_resized, face1_cropped_resized, face2_cropped_resized;
    cv::resize(img1, img1_resized, cv::Size(), 2, 2, cv::INTER_AREA);
    cv::resize(img2, img2_resized, cv::Size(), 2, 2, cv::INTER_AREA);
    cv::resize(src1_mask, src_mask1_resized, cv::Size(), 2, 2, cv::INTER_AREA);
    cv::resize(src2_mask, src_mask2_resized, cv::Size(), 2, 2, cv::INTER_AREA);
    cv::resize(face1_cropped, face1_cropped_resized, cv::Size(), 2, 2, cv::INTER_AREA);
    cv::resize(face2_cropped, face2_cropped_resized, cv::Size(), 2, 2, cv::INTER_AREA);
    
    // resize crop and mask images based on ratios of face sizes:
    float x_ratio_face12 = rect1.width / rect2.width;
    float y_ratio_face12 = rect1.height / rect2.height;
    cv::resize(src_mask1_resized, src_mask1_resized, cv::Size(), 1 / 1, 1 / 1, cv::INTER_AREA);
    cv::resize(face1_cropped_resized, face1_cropped_resized, cv::Size(), 1, 1, cv::INTER_AREA);
    cv::resize(src_mask2_resized, src_mask2_resized, cv::Size(), x_ratio_face12, y_ratio_face12, cv::INTER_AREA);
    cv::resize(face2_cropped_resized, face2_cropped_resized, cv::Size(), x_ratio_face12, y_ratio_face12, cv::INTER_AREA);
    cv::resize(img2_resized, img2_resized, cv::Size(), x_ratio_face12, y_ratio_face12, cv::INTER_AREA);

    // swap faces:
    // int centre_x_f1 = int((face_rect1[0][2].x - face_rect1[0][0].x) / 2);
    // int centre_y_f1 = int((face_rect1[0][2].y - face_rect1[0][0].y) / 2) ;
    int centre_x_f1 = rect1.x + int((rect1.width) / 2);
    int centre_y_f1 = rect1.y + int((rect1.height) / 2);
    cv::Point centre_face1 = cv::Point(centre_x_f1 * 2, centre_y_f1 * 2);

    int centre_x_f2 = rect2.x + int((rect2.width) / 2);
    int centre_y_f2 = rect2.y + int((rect2.height) / 2);
    cv::Point centre_face2 = cv::Point(centre_x_f2 * 2 * x_ratio_face12, centre_y_f2 * 2 * y_ratio_face12);

   

    cv::Mat swap12, swap21;
    std::cout << "\n\nface2_cropped_resized.size(): " << face2_cropped_resized.size() << "\nsrc_mask2_resized.size(): " <<
        src_mask2_resized.size() << "\img1_resized.size(): " <<
        img1_resized.size();

    std::cout << "\n\face1_cropped_resized.size(): " << face1_cropped_resized.size() << "\src_mask1_resized.size(): " <<
        src_mask1_resized.size() << "\img2_resized.size(): " <<  img2_resized.size();

    seamlessClone(face2_cropped_resized, img1_resized, src_mask2_resized, centre_face1, swap12, MIXED_CLONE);
    seamlessClone(face1_cropped_resized, img2_resized, src_mask1_resized, centre_face2, swap21, MIXED_CLONE);
    cv::Mat swap12_resized, swap21_resized;
    cv::resize(swap12, swap12_resized, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::resize(swap21, swap21_resized, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    
    cv::circle(img1_resized, centre_face1, 30, Scalar(0, 0, 255), FILLED, LINE_8);
    cv::resize(img1_resized, img1_resized, cv::Size(), 0.25, 0.25, cv::INTER_AREA);

    cv::circle(img2_resized, centre_face2, 30, Scalar(0, 0, 255), FILLED, LINE_8);
    cv::resize(img2_resized, img2_resized, cv::Size(), 0.25, 0.25, cv::INTER_AREA);
    cv::imshow("img1_resized", img1_resized);
    cv::imshow("img2_resized", img2_resized);
    cv::imshow("swap12_resized", swap12_resized); 
    cv::imshow("swap21_resized", swap21_resized);
    cv::waitKey();
    cv::imwrite("../../data/michael_jordan_as_theresa_may.jpg", img1_resized);
    cv::imwrite("../../data/theresa_may_as_michael_jordan.jpg", img2_resized);

}