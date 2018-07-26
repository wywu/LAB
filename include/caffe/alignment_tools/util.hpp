#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include <vector>
#include <string>
#include <mutex>
#include <cstdio>
#include <functional>

#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>

#include "caffe/alignment_tools/io.hpp"

namespace cv {
	typedef Rect_<float> Rect2f;
} // namespace cv

namespace alignment_tools {


class SafeCounter {
public:
	SafeCounter(int total=0, int output=0): m_total(total), m_output(output), m_count(0) {}
	inline void Reset(void) {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_count=0;
	}
	inline int operator++() {
		std::lock_guard<std::mutex> lock(m_mutex);
		m_count++;
		int ret=m_count;
		if(m_count%m_output==0) {
			fprintf(stderr, "Counter: %d of %d.\n", m_count, m_total);
		}
		return ret;
	}
	inline int operator++(int) {
		std::lock_guard<std::mutex> lock(m_mutex);
		int ret=m_count;
		m_count++;
		if(m_count%m_output==0) {
			fprintf(stderr, "Counter: %d of %d.\n", m_count, m_total);
		}
		return ret;
	}

private:
	std::mutex m_mutex;
	int m_count;
	int m_total;
	int m_output;
};

template <typename T>
std::vector<T> Filter(const std::vector<T>& src, const std::vector<bool>& filter) {
	CHECK_EQ(src.size(), filter.size());
	std::vector<T> ret;
	for(size_t i=0; i<src.size(); i++) {
		if(filter[i]) {
			ret.push_back(src[i]);
		}
	}
	return ret;
}

template <typename T>
inline std::vector<T> ToVector(const caffe::Blob<T>& blob) {
	std::vector<T> vec(blob.count());
	memcpy(vec.data(), blob.cpu_data(), vec.size()*sizeof(T));
	return vec;
}

inline std::vector<cv::Point2f> ToPoints(const std::vector<float>& labels) {
	CHECK_EQ(labels.size()%2, 0);
	std::vector<cv::Point2f> points(labels.size()/2);
	for(size_t i=0; i<points.size(); i++) {
		points[i].x=labels[2*i];
		points[i].y=labels[2*i+1];
	}
	return points;
}

inline std::vector<float> ToLabel(const std::vector<cv::Point2f>& points) {
	std::vector<float> label(points.size()*2);
	for(size_t i=0; i<points.size(); i++) {
		label[2*i]=points[i].x;
		label[2*i+1]=points[i].y;
	}
	return label;
}

inline void Copy(const cv::Mat& cv_mat, caffe::Blob<float>& blob) {
	CHECK_EQ(cv_mat.depth(), CV_32F);
	blob.Reshape(1, cv_mat.channels(), cv_mat.rows, cv_mat.cols);
	if(cv_mat.channels()==1) {
		memcpy(blob.mutable_cpu_data(), cv_mat.data, blob.count()*sizeof(float));
	}
	else {
		std::vector<cv::Mat> splited_img;
		cv::split(cv_mat, splited_img);
		for(size_t i=0; i<splited_img.size(); i++) {
			memcpy(blob.mutable_cpu_data()+blob.count(2)*i, splited_img[i].data, blob.count(2)*sizeof(float));
		}
	}
}

inline void Copy(const std::vector<cv::Mat>& cv_mat, caffe::Blob<float>& blob) {
	CHECK_GT(cv_mat.size(), 0);
	for(size_t i=0; i<cv_mat.size(); i++) {
		CHECK_EQ(cv_mat[i].depth(), CV_32F);
		CHECK_EQ(cv_mat[i].rows, cv_mat[0].rows);
		CHECK_EQ(cv_mat[i].cols, cv_mat[0].cols);
		CHECK_EQ(cv_mat[i].channels(), cv_mat[0].channels());
	}
	blob.Reshape(cv_mat.size(), cv_mat[0].channels(), cv_mat[0].rows, cv_mat[0].cols);
	if(cv_mat[0].channels()==1) {
		for(size_t i=0; i<cv_mat.size(); i++) {
			memcpy(blob.mutable_cpu_data()+blob.count(1)*i, cv_mat[i].data, blob.count(1)*sizeof(float));
		}
	}
	else {
		for(size_t i=0; i<cv_mat.size(); i++) {
			std::vector<cv::Mat> splited_img;
			cv::split(cv_mat[i], splited_img);
			for(size_t j=0; j<splited_img.size(); j++) {
				memcpy(blob.mutable_cpu_data()+blob.count(1)*i+blob.count(2)*j, splited_img[j].data, blob.count(2)*sizeof(float));
			}
		}
	}
}

void ConvertImageToGray(cv::Mat &image);
void ConvertImageToBGR(cv::Mat &image);
void NormalizeImage(cv::Mat &img);
std::vector<cv::Point2f> InvAffinePose(const cv::Mat_<float>& affine_mat, const std::vector<cv::Point2f>& pose);
cv::Mat_<float> CalcAffineMatByPose(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst);

} // namespace alignment_tools

#endif // #ifndef __UTIL_HPP__
