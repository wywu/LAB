#include <cstdio>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/alignment_tools/util.hpp"
#include "caffe/alignment_tools/io.hpp"

using namespace std;
using namespace cv;

namespace alignment_tools {


void ConvertImageToGray(Mat &image) {
	if(image.channels()==1) {
		return;
	}
	else if(image.channels()==4) {
		cvtColor(image, image, CV_BGRA2GRAY);
		return;
	}
	else if(image.channels()==3) {
		cvtColor(image, image, CV_BGR2GRAY);
		return;
	}
	else {
		LOG(FATAL) <<"Convert image to Gray failed" << endl;
	}
}

void ConvertImageToBGR(Mat &image) {
	if(image.channels()==1) {
		cvtColor(image, image, CV_GRAY2BGR);
		return;
	}
	else if(image.channels()==4) {
		cvtColor(image, image, CV_BGRA2BGR);
		return;
	}
	else if(image.channels()==3) {
		return;
	}
	else {
		LOG(FATAL) <<"Convert image to BGR failed" << endl;
	}
}

// calculate a affine matrix by src to dst pose
cv::Mat_<float> CalcAffineMatByPose(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst) {
	int point_num=src.size();

	Mat_<float> X(2*point_num, 4);
	Mat_<float> U(2*point_num, 1);
	for (unsigned int i=0;i<point_num;i++) {
		X(i, 0)=src[i].x;
		X(i+point_num, 0)=src[i].y;
		X(i, 1)=src[i].y;
		X(i+point_num, 1)=-1*src[i].x;
		X(i, 2)=1;
		X(i+point_num, 3)=1;
		X(i, 3)=0;
		X(i+point_num, 2)=0;

		U(i, 0)=dst[i].x;
		U(i+point_num, 0)=dst[i].y;
	}
	//X.inv(DECOMP_SVD): compute pseudo-inverse of X
	Mat_<float> result=X.inv(DECOMP_SVD)*U;

	Mat_<float> affine_mat(2, 3);
	affine_mat(0, 0)=result(0, 0);
	affine_mat(0, 1)=result(1, 0);
	affine_mat(0, 2)=result(2, 0);
	affine_mat(1, 0)=-1*result(1, 0);
	affine_mat(1, 1)=result(0, 0);
	affine_mat(1, 2)=result(3, 0);
	return affine_mat;
}

void NormalizeImage(Mat &img) {
	Mat mean;
	Mat std;
	meanStdDev(img, mean, std);
	for(size_t i=0;i<img.channels();i++) {
		if(std.at<double>(i, 0)<1E-6) {
			std.at<double>(i, 0)=1;
		}
	}

	vector<Mat> split_img(img.channels());
	split(img, &(split_img[0]));
	for(size_t i=0;i<img.channels();i++) {
		split_img[i].convertTo(split_img[i], CV_32F, 1.0/std.at<double>(i, 0), -1*mean.at<double>(i, 0)/std.at<double>(i, 0));
	}
	merge(&(split_img[0]), img.channels(), img);
}

std::vector<cv::Point2f> InvAffinePose(const cv::Mat_<float>& affine_mat, const std::vector<cv::Point2f>& pose) {
	vector<Point2f> inv_pose(pose.size());
	for(size_t i=0; i<inv_pose.size(); i++) {
		inv_pose[i].x=pose[i].x-affine_mat(0, 2);
		inv_pose[i].y=pose[i].y-affine_mat(1, 2);
	}
	float scale=affine_mat(0, 0)*affine_mat(0, 0)+affine_mat(0, 1)*affine_mat(0, 1);
	float inv_00=affine_mat(0, 0)/scale;
	float inv_01=-1*affine_mat(0, 1)/scale;
	float inv_10=-1*inv_01;
	float inv_11=inv_00;
	for(size_t i=0; i<inv_pose.size(); i++) {
		inv_pose[i]=Point2f(inv_00*inv_pose[i].x+inv_01*inv_pose[i].y,
				inv_10*inv_pose[i].x+inv_11*inv_pose[i].y);
	}
	return inv_pose;
}

} // namespace alignment_tools
