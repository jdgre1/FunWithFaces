#include <face_library.h> 
#include <fstream>
using namespace cv;
using namespace std;


FaceLibrary::FaceLibrary() {

	/*m_api = new tesseract::TessBaseAPI();
	if (m_api->Init(NULL, "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}
	m_api->SetDebugVariable("debug_file", "/dev/null");*/
}

FaceLibrary::~FaceLibrary() {}

std::string FaceLibrary::getModelTxt() const {
	return m_modelTxt;;
}
std::string FaceLibrary::getModel() const {
	return m_modelBin;;
}

std::vector<cv::Rect> faces;

// Credit: https://bewagner.net/programming/2020/04/12/building-a-face-detector-with-opencv-in-cpp/
cv::Rect find_best_face_position(cv::Mat& detection_matrix, cv::Mat& input_img) {

    float max_confidence_threshold_ = 0.165;
    int max_area = 0.;
    cv::Rect face;

    for (int i = 0; i < detection_matrix.rows; i++) {
        float confidence = detection_matrix.at<float>(i, 2);

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

cv::Rect FaceLibrary::return_facebox(cv::Mat& img) {

    int img_height = img.rows;
    int img_width = img.cols;
    cv::Mat blob_input;

    // Set input blob for detections
    cv::resize(img, blob_input, cv::Size(), 300, 300);
    cv::Mat blob = cv::dnn::blobFromImage(blob_input, 1.0, Size(300, 300), (104.0, 177.0, 123.0));
    m_dnnNet.setInput(blob);

    // forward pass of neural network
    //std::vector<cv::Mat> detections = m_dnnNet.forward();
    cv::Mat detections = m_dnnNet.forward();
   
    cv::Rect facebox = find_best_face_position(detections, img);
    return facebox;
}

void FaceLibrary::setDnnNet(const cv::dnn::Net& net)
{
	//cout << net.empty() << endl;
	m_dnnNet = net;
}

