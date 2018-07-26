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

template<typename T>
std::string ToString(const std::string& format, T data) {
	char buf[max_line];
	sprintf(buf, format.c_str(), data);
	return std::string(buf);
}

inline std::string StringReplace(std::string src,
		const std::string& mode, const std::string& re_mode) {
	for(;;) {
		size_t pos=src.find(mode);
		if(pos==std::string::npos) {
			break;
		}
		else {
			src.replace(pos, mode.size(), re_mode);
		}
	}
	return src;
}

template<typename T>
inline int GetDigit(T num) {
	int digit=0;
	while(num>0) {
		num/=10;
		digit++;
	}
	return digit;
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

inline std::vector<cv::Point2d> ToPointsD(const std::vector<double>& labels) {
	CHECK_EQ(labels.size()%2, 0);
	std::vector<cv::Point2d> points(labels.size()/2);
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

inline std::vector<double> ToLabelD(const std::vector<cv::Point2d>& points) {
	std::vector<double> label(points.size()*2);
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

inline void CopyLabel(const cv::Mat& cv_mat, caffe::Blob<float>& blob) {
	CHECK_EQ(cv_mat.depth(), CV_32F);
	blob.Reshape(1, cv_mat.rows, 1, 1);
	memcpy(blob.mutable_cpu_data(), cv_mat.data, blob.count()*sizeof(float));
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

template <typename T>
inline std::vector<T> ToVector(const caffe::Blob<T>& blob) {
	std::vector<T> vec(blob.count());
	memcpy(vec.data(), blob.cpu_data(), vec.size()*sizeof(T));
	return vec;
}

template <typename T>
inline std::vector<std::vector<T>> ToVector2D(const caffe::Blob<T>& blob) {
	std::vector<std::vector<T>> vec2d(blob.num());
	size_t dim2=blob.count(1);
	for(size_t i=0; i<vec2d.size(); i++) {
		vec2d[i].resize(dim2);
		memcpy(vec2d[i].data(), blob.cpu_data()+dim2*i, dim2*sizeof(T));
	}
	return vec2d;
}

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
size_t FindMaxIndex(const std::vector<T>& list, const std::function<bool(const T&, const T&)>& bigger) {
	CHECK_GT(list.size(), 0);
	size_t max_index=0;
	for(size_t i=1; i<list.size(); i++) {
		if(bigger(list[i], list[max_index])) {
			max_index=i;
		}
	}
	return max_index;
}

template <typename T>
std::vector<T> Inverse(const std::vector<T>& list) {
	std::vector<T> inv_list(list.size());
	for(size_t i=0; i<inv_list.size(); i++) {
		inv_list[i]=list[list.size()-1-i];
	}
	return inv_list;
}

std::string RandomNumberString(int length);

void ConvertImageToBGR(cv::Mat &image);
void ConvertImageToGray(cv::Mat &image);
void ConvertImageToBGRA(cv::Mat &image);
void DrawPointsOnImage(cv::Mat &image, const std::vector<float> &points);
void DrawRectsOnImage(cv::Mat &image, const std::vector<float> &rects);
void SafeMatCrop(const cv::Mat &image, cv::Rect rect, cv::Mat &roi);

std::vector<cv::Point2f> LandmarkChangeTo21p(const std::vector<cv::Point2f> landmark);
cv::Rect PoseToRect(const std::vector<cv::Point2f> &pose);

void CropImageWithPadding(const cv::Mat& img, cv::Rect rect, float padding_ratio, cv::Size size_output, cv::Mat& cropped_img);
void CropPointWithPadding(const cv::Mat& img, cv::Rect rect, float padding_ratio, cv::Size size_output, std::vector<cv::Point2f>& lanmdmark);
void InvCropPointWithPadding(const cv::Size& img_size, cv::Rect rect, float padding_ratio, cv::Size size_output, std::vector<cv::Point2f>& lanmdmark);
void InvCropPointWithPadding(const cv::Mat& img, cv::Rect rect, float padding_ratio, cv::Size size_output, std::vector<cv::Point2f>& lanmdmark);

cv::Mat_<float> CalcAffineMatByPose(const std::vector<cv::Point2f> &src, const std::vector<cv::Point2f> &dst);
std::vector<cv::Point2f> AffinePose(const cv::Mat_<float>& affine_mat, const std::vector<cv::Point2f>& pose);
std::vector<cv::Point2f> InvAffinePose(const cv::Mat_<float>& affine_mat, const std::vector<cv::Point2f>& pose);
std::vector<cv::Point2d> InvAffinePoseD(const cv::Mat_<double>& affine_mat, const std::vector<cv::Point2d>& pose);
void NormalizeImage(cv::Mat &img);

std::vector<cv::Rect2f> LabelToRect(const std::vector<float> &label);

float GetMeanValue(const std::vector<float>& data);
float IoU(const cv::Rect& r1, const cv::Rect& r2);
float Distance(const cv::Point2f& p1, const cv::Point2f& p2);
std::vector<cv::Point2f> CurveInterp(const std::vector<cv::Point2f>& src, int samples);

bool Is106ptValid(const std::vector<cv::Point2f>& landmarks);
bool Is106ptValidProfile(const std::vector<cv::Point2f>& landmarks);

bool IsValid(const Label_2016& label);
bool IsValidProfile(const Label_2016& label);
bool IsLeftToRight(const std::vector<cv::Point2f>& landmarks);
bool IsTopToBottom(const std::vector<cv::Point2f>& landmarks);
std::vector<cv::Point2f> KeepLeftToRight(const std::vector<cv::Point2f>& src, const cv::Mat_<float>& affine_mat);
std::vector<cv::Point2f> KeepTopToBottom(const std::vector<cv::Point2f>& src, const cv::Mat_<float>& affine_mat);
std::vector<cv::Point2f> To106pt(Label_2016 label);
std::vector<cv::Point2f> To106ptProfile(Label_2016 label);
std::vector<cv::Point2f> To68pt(const std::vector<cv::Point2f>& landmarks);
std::vector<cv::Point2f> ToCheckable106pt(Label_2016 label);
Label_2016 ToLabel2016(const std::vector<cv::Point2f>& label);

void randomParaGenerator(cv::Mat image, std::vector<cv::Mat>& augmented_img, std::vector<cv::Mat>& augmented_trans, int sample_num);
bool sort_min (float i, float j);

} // namespace alignment_tools

#endif // #ifndef __UTIL_HPP__
