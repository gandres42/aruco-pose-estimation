#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap = cv::VideoCapture(0);
    cv::Mat frame;
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    while (true) {
        // detect markers in frame
        cap.read(frame);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cv::aruco::detectMarkers(frame, dictionary, corners, ids);

        // annotate any detected markers 
        cv::Mat frame_copy;
        frame.copyTo(frame_copy);
        if (ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(frame_copy, corners, ids);
        }

        // display detected tags
        cv::imshow("detected tags", frame_copy);
        char key = (char) cv::waitKey(1);
        if (key == 27) {
            break;
        }

        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
            1000, 0, frame.cols / 2,
            0, 1000, frame.rows / 2,
            0, 0, 1
        );

        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5) << 
            0, 0, 0, 0, 0
        );
        
        // You can read camera parameters from tutorial_camera_params.yml
        // readCameraParameters(filename, cameraMatrix, distCoeffs); // This function is located in detect_markers.cpp
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
    }
    return 0;
}
