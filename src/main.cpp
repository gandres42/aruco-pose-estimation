#include <iostream>
#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include "sensor_msgs/Image.h"
#include <sensor_msgs/image_encodings.h>
#include "geometry_msgs/PoseStamped.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Dense>

struct TagInfo {
    float size;
};

struct Poses {
    cv::Vec3d rvec;
    cv::Vec3d tvec;
};

class Aruco {
private:
    // icky ROS stuff
    ros::NodeHandle node;
    ros::Subscriber camera_sub;
    ros::Publisher pose_pub;

    // aruco detection
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Mat cameraMatrix, distCoeffs;
    std::map<int, Poses> prev_poses;

    // tag data
    std::map<int, TagInfo> tag_info;
  
public:
    Aruco() {
        camera_sub = node.subscribe<sensor_msgs::Image>("BlueROV2/video", 1, &Aruco::camera_cb, this);
        pose_pub = node.advertise<geometry_msgs::PoseStamped>("/aruco/pose", 1);
        dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        prev_poses = std::map<int, Poses>();

        // camera calibration
        cameraMatrix = (cv::Mat_<double>(3, 3) << 
            1078.17559, 0.0, 1010.57086,
            0.0, 1076.46176, 463.06243,
            0.0, 0.0, 1.0);
        distCoeffs = (cv::Mat_<double>(1, 5) << 0.019645, 0.007271, -0.004324, -0.001628, 0.000000);

        // tag data
        tag_info = {
            {4, {.15}}
        };
    }

    void camera_cb(const sensor_msgs::Image::ConstPtr& msg) {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

        // detect markers
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::detectMarkers(cv_ptr->image, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);

        // estimate marker poses using custom solver
        std::vector<cv::Vec3d> rvecs, tvecs;
        rvecs.resize(markerIds.size());
        tvecs.resize(markerIds.size());
        
        // Estimate pose for each detected marker using SOLVEPNP_IPPE_SQUARE solver
        for (int i = 0; i < markerIds.size(); i++) {
            // skip tags we don't know about
            if (this->tag_info.find(markerIds[i]) == this->tag_info.end()) {
                continue;
            }

            // generate real marker points, assume it's at 0, 0, 0 for now
            float marker_size = this->tag_info[markerIds[i]].size;
            std::vector<cv::Point3f> objectPoints = {
                cv::Point3f(-marker_size/2,  marker_size/2, 0),
                cv::Point3f( marker_size/2,  marker_size/2, 0),
                cv::Point3f( marker_size/2, -marker_size/2, 0),
                cv::Point3f(-marker_size/2, -marker_size/2, 0)
            };

            // use previous pose estimate if available for ID
            if (this->prev_poses.find(markerIds[i]) != this->prev_poses.end()) {
                rvecs[i] = this->prev_poses[markerIds[i]].rvec;
                tvecs[i] = this->prev_poses[markerIds[i]].tvec;
            }
            cv::solvePnP(objectPoints, markerCorners[i], cameraMatrix, distCoeffs, rvecs[i], tvecs[i], true, cv::SOLVEPNP_ITERATIVE);
            this->prev_poses[markerIds[i]] = {rvecs[i], tvecs[i]};

            // Convert rvec to rotation matrix
            cv::Mat R;
            cv::Rodrigues(rvecs[i], R);

            // Invert the transform: tag->camera
            cv::Mat R_inv = R.t();
            cv::Mat tvec_mat = cv::Mat(tvecs[i]);
            cv::Mat tvec_inv = -R_inv * tvec_mat;

            // Convert rotation matrix back to rvec
            cv::Vec3d rvec_inv;
            cv::Rodrigues(R_inv, rvec_inv);

            std::cout << "Rotation Vector (rvec, tag->camera): " << rvec_inv << std::endl;
            std::cout << "Translation Vector (tvec, tag->camera): " << tvec_inv.t() << std::endl;

            geometry_msgs::PoseStamped pose_msg;
            pose_msg.header.stamp = ros::Time::now();
            pose_msg.header.frame_id = "map";

            // Set position (translation)
            pose_msg.pose.position.x = tvec_inv.at<double>(0);
            pose_msg.pose.position.y = tvec_inv.at<double>(1);
            pose_msg.pose.position.z = tvec_inv.at<double>(2);

            // Convert rvec_inv to quaternion
            cv::Mat R_inv_mat;
            cv::Rodrigues(rvec_inv, R_inv_mat);
            Eigen::Matrix3d eig_R;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    eig_R(row, col) = R_inv_mat.at<double>(row, col);
            Eigen::Quaterniond q(eig_R);

            pose_msg.pose.orientation.x = q.x();
            pose_msg.pose.orientation.y = q.y();
            pose_msg.pose.orientation.z = q.z();
            pose_msg.pose.orientation.w = q.w();

            this->pose_pub.publish(pose_msg);
        }

        // visualize markers and poses
        cv::Mat outputImage = cv_ptr->image.clone();
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
        for (int i = 0; i < rvecs.size(); ++i) {
            if (this->tag_info.find(markerIds[i]) == this->tag_info.end()) {
                continue;
            }
            auto rvec = rvecs[i];
            auto tvec = tvecs[i];
            cv::aruco::drawAxis(outputImage, cameraMatrix, distCoeffs, rvec, tvec, 0.1);
        }
        cv::resize(outputImage, outputImage, cv::Size(outputImage.cols / 2, outputImage.rows / 2));
        cv::imshow("camera", outputImage);
        cv::waitKey(1);
    }
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "aruco");
    Aruco aruco;
    ros::spin();
    return 0;
}