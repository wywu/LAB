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

std::string RandomNumberString(int length) {
	CHECK_GE(length, 0);
	string str;
	str.resize(length);
	random_device rd;
	mt19937 engine(rd());
	uniform_int_distribution<int> rng('0', '9');
	for(auto& c: str) {
		c=rng(engine);
	}
	return str;
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

void ConvertImageToBGRA(Mat &image) {
	if(image.channels()==1) {
		cvtColor(image, image, CV_GRAY2BGRA);
		return;
	}
	else if(image.channels()==4) {
		return;
	}
	else if(image.channels()==3) {
		cvtColor(image, image, CV_BGR2BGRA);
		return;
	}
	else {
		LOG(FATAL) <<"Convert image to BGRA failed" << endl;
	}
}

void DrawPointsOnImage(Mat &image, const vector<float> &points) {
	CHECK_EQ(points.size()%2, 0);
	for(size_t i=0;i<points.size()/2;i++) {
		circle(image, Point2f(points[i*2], points[i*2+1]), 2, Scalar::all(255), 2);
	}
	return;
}

// void DrawRectsOnImage(Mat &image, const vector<float> &rects) {
// 	CHECK_EQ(rects.size()%5, 0);
// 	for(size_t i=0;i<rects.size()/5;i++) {
// 		if(rects[i*5+0]<0) {
// 			continue;
// 		}
// 		rectangle(image, Rect(rects[i*5+1], rects[i*5+2], rects[i*5+3]-rects[i*5+1]+1, rects[i*5+4]-rects[i*5+2]+1), Scalar::all(255), 2);
// 	}
// 	return;
// }

// @Wenyan Wu 2017/03/25
// change rect size to 4
void DrawRectsOnImage(Mat &image, const vector<float> &rects) {
	CHECK_EQ(rects.size()%4, 0);
	for(size_t i=0;i<rects.size()/4;i++) {
		rectangle(image, Rect(rects[i*4], rects[i*4+1], rects[i*4+2]-rects[i*4]+1, rects[i*4+3]-rects[i*4+1]+1), Scalar::all(255), 2);
	}
	return;
}

//calculate the face's rectangle using it's pose
Rect PoseToRect(const vector<Point2f> &pose) {
	vector<Point2f> pose_21;
	if(pose.size()!=21) {
		pose_21=LandmarkChangeTo21p(pose);
	}
	else {
		pose_21=pose;
	}
	CHECK_EQ(pose_21.size(), 21);

	vector<float> s_distance;
	for (size_t i = 0; i<pose_21.size() - 1; i++) {
		for (size_t j = i + 1; j<pose_21.size(); j++) {
			s_distance.push_back((pose_21[i].x - pose_21[j].x)*(pose_21[i].x - pose_21[j].x) + (pose_21[i].y - pose_21[j].y)*(pose_21[i].y - pose_21[j].y));
		}
	}
	sort(s_distance.begin(), s_distance.end());
	float size = sqrt(s_distance[s_distance.size()-1]) / 2;

	float center_x = (pose_21[10].x + pose_21[11].x + pose_21[12].x + pose_21[18].x) / 4;
	float center_y = (pose_21[10].y + pose_21[11].y + pose_21[12].y + pose_21[18].y) / 4;

	Rect rect;
	rect.x = center_x - size;
	rect.width = 2*size;
	rect.y = center_y - size;
	rect.height = 2*size;

	return rect;
}

vector<Point2f> LandmarkChangeTo21p(const vector<Point2f> landmark) {
	vector<Point2f> landmark_21;
	if (landmark.size() == 68) {
		landmark_21.push_back(landmark[17]);
		landmark_21.push_back(landmark[19]);
		landmark_21.push_back(landmark[21]);

		landmark_21.push_back(landmark[22]);
		landmark_21.push_back(landmark[24]);
		landmark_21.push_back(landmark[26]);

		landmark_21.push_back(landmark[36]);
		landmark_21.push_back(landmark[39]);

		landmark_21.push_back(landmark[42]);
		landmark_21.push_back(landmark[45]);

		landmark_21.push_back(landmark[31]);
		landmark_21.push_back(landmark[33]);
		landmark_21.push_back(landmark[35]);

		landmark_21.push_back(landmark[51]);
		landmark_21.push_back(Point2f((landmark[51].x + landmark[57].x) / 2, (landmark[51].y + landmark[57].y) / 2));
		landmark_21.push_back(landmark[57]);

		landmark_21.push_back(Point2f((landmark[36].x + landmark[39].x) / 2, (landmark[36].y + landmark[39].y) / 2));
		landmark_21.push_back(Point2f((landmark[42].x + landmark[45].x) / 2, (landmark[42].y + landmark[45].y) / 2));
		landmark_21.push_back(landmark[30]);
		landmark_21.push_back(landmark[48]);
		landmark_21.push_back(landmark[54]);
	}
	else if(landmark.size() == 194) {
		landmark_21.push_back(landmark[184]);
		landmark_21.push_back(landmark[179]);
		landmark_21.push_back(landmark[174]);

		landmark_21.push_back(landmark[154]);
		landmark_21.push_back(landmark[162]);
		landmark_21.push_back(landmark[165]);

		landmark_21.push_back(landmark[144]);
		landmark_21.push_back(landmark[134]);

		landmark_21.push_back(landmark[115]);
		landmark_21.push_back(landmark[125]);

		landmark_21.push_back(landmark[45]);
		landmark_21.push_back(landmark[49]);
		landmark_21.push_back(landmark[53]);

		landmark_21.push_back(landmark[65]);
		landmark_21.push_back(landmark[92]);
		landmark_21.push_back(landmark[81]);

		landmark_21.push_back(landmark[139]);
		landmark_21.push_back(landmark[121]);
		Point2f nose = (landmark[41] + landmark[46] + landmark[46] + landmark[57]);
		nose.x /= 4;
		nose.y /= 4;
		landmark_21.push_back(nose);
		landmark_21.push_back(landmark[58]);
		landmark_21.push_back(landmark[72]);
	}
	else if(landmark.size() == 106) {
		landmark_21.push_back(landmark[33]);
		Point2f eyebrow_l_1 = landmark[35] + landmark[65];
		eyebrow_l_1.x /= 2;
		eyebrow_l_1.y /= 2;
		landmark_21.push_back(eyebrow_l_1);
		Point2f eyebrow_l_2 = landmark[37] + landmark[67];
		eyebrow_l_2.x /= 2;
		eyebrow_l_2.y /= 2;
		landmark_21.push_back(eyebrow_l_2);

		Point2f eyebrow_r_0 = landmark[38] + landmark[68];
		eyebrow_r_0.x /= 2;
		eyebrow_r_0.y /= 2;
		landmark_21.push_back(eyebrow_r_0);
		Point2f eyebrow_r_1 = landmark[40] + landmark[70];
		eyebrow_r_1.x /= 2;
		eyebrow_r_1.y /= 2;
		landmark_21.push_back(eyebrow_r_1);
		landmark_21.push_back(landmark[42]);

		landmark_21.push_back(landmark[52]);
		landmark_21.push_back(landmark[55]);

		landmark_21.push_back(landmark[58]);
		landmark_21.push_back(landmark[61]);

		landmark_21.push_back(landmark[82]);
		landmark_21.push_back(landmark[49]);
		landmark_21.push_back(landmark[83]);

		landmark_21.push_back(landmark[87]);
		Point2f mouth_c = landmark[98] + landmark[102];
		mouth_c.x /= 2;
		mouth_c.y /= 2;
		landmark_21.push_back(mouth_c);
		landmark_21.push_back(landmark[93]);

		landmark_21.push_back(landmark[104]);
		landmark_21.push_back(landmark[105]);
		landmark_21.push_back(landmark[46]);
		landmark_21.push_back(landmark[84]);
		landmark_21.push_back(landmark[90]);
	}
	else if(landmark.size() == 91) {
		landmark_21.push_back(landmark[17]);
		Point2f eyebrow_l_1 = landmark[19] + landmark[21];
		eyebrow_l_1.x /= 2;
		eyebrow_l_1.y /= 2;
		landmark_21.push_back(eyebrow_l_1);
		Point2f eyebrow_l_2 = landmark[21] + landmark[51];
		eyebrow_l_2.x /= 2;
		eyebrow_l_2.y /= 2;
		landmark_21.push_back(eyebrow_l_2);

		Point2f eyebrow_r_0 = landmark[22] + landmark[52];
		eyebrow_r_0.x /= 2;
		eyebrow_r_0.y /= 2;
		landmark_21.push_back(eyebrow_r_0);
		Point2f eyebrow_r_1 = landmark[24] + landmark[54];
		eyebrow_r_1.x /= 2;
		eyebrow_r_1.y /= 2;
		landmark_21.push_back(eyebrow_r_1);
		landmark_21.push_back(landmark[26]);

		landmark_21.push_back(landmark[36]);
		landmark_21.push_back(landmark[39]);

		landmark_21.push_back(landmark[42]);
		landmark_21.push_back(landmark[45]);

		landmark_21.push_back(landmark[66]);
		landmark_21.push_back(landmark[33]);
		landmark_21.push_back(landmark[67]);

		landmark_21.push_back(landmark[74]);
		Point2f mouth_c = landmark[85] + landmark[89];
		mouth_c.x /= 2;
		mouth_c.y /= 2;
		landmark_21.push_back(mouth_c);
		landmark_21.push_back(landmark[80]);

		Point2f left_eye_c = landmark[36]+landmark[37]+landmark[56]+landmark[38]+landmark[39]+landmark[40]+
				landmark[57]+landmark[41];
		left_eye_c.x/=8.0f;
		left_eye_c.y/=8.0f;
		landmark_21.push_back(left_eye_c);
		Point2f right_eye_c = landmark[42]+landmark[43]+landmark[59]+landmark[44]+landmark[45]+landmark[46]+
				landmark[60]+landmark[47];
		right_eye_c.x/=8.0f;
		right_eye_c.y/=8.0f;
		landmark_21.push_back(right_eye_c);
		landmark_21.push_back(landmark[30]);
		landmark_21.push_back(landmark[71]);
		landmark_21.push_back(landmark[77]);
	}
	else if(landmark.size() == 74) {
		landmark_21.resize(21);
		landmark_21[0] = landmark[21];
		landmark_21[1].x = ( landmark[22].x + landmark[23].x + landmark[25].x + landmark[26].x ) / 4;
		landmark_21[1].y = ( landmark[22].y + landmark[23].y + landmark[25].y + landmark[26].y ) / 4;
		landmark_21[2] = landmark[24];
		landmark_21[3] = landmark[18];
		landmark_21[4].x = ( landmark[16].x + landmark[17].x + landmark[19].x + landmark[20].x ) / 4;
		landmark_21[4].y = ( landmark[16].y + landmark[17].y + landmark[19].y + landmark[20].y ) / 4;
		landmark_21[5] = landmark[15];
		landmark_21[6] = landmark[27];
		landmark_21[7] = landmark[29];
		landmark_21[8] = landmark[33];
		landmark_21[9] = landmark[31];
		landmark_21[10] = landmark[38];
		landmark_21[11] = landmark[39];
		landmark_21[12] = landmark[40];
		landmark_21[13] = landmark[49];
		landmark_21[14] = landmark[64];
		landmark_21[15] = landmark[55];
		landmark_21[16].x = (landmark[28].x + landmark[30].x) / 2;
		landmark_21[16].y = (landmark[28].y + landmark[30].y) / 2;
		landmark_21[17].x = (landmark[32].x + landmark[34].x) / 2;
		landmark_21[17].y = (landmark[32].y + landmark[34].y) / 2;
		landmark_21[18] = landmark[65];
		landmark_21[19] = landmark[46];
		landmark_21[20] = landmark[52];
	}
	else if(landmark.size() == 21) {
		landmark_21=landmark;
	}

	return landmark_21;
}

//crop a image with padding ratio
void CropImageWithPadding(const Mat& img, Rect rect, float padding_ratio, Size size_output, Mat& cropped_img) {
	//padding
	Rect rect_padded=rect;
	rect_padded.x-=(padding_ratio-1)*(rect.width)/2.0;
	rect_padded.width=rect_padded.width*padding_ratio;
	rect_padded.y-=(padding_ratio-1)*(rect.height)/2.0;
	rect_padded.height=rect_padded.height*padding_ratio;
	rect=rect_padded;

	// crop
	SafeMatCrop(img, rect, cropped_img);
	resize(cropped_img, cropped_img, size_output);
}

//crop a image with padding ratio
void CropPointWithPadding(const Mat& img, Rect rect, float padding_ratio, Size size_output, vector<Point2f>& lanmdmark) {
	//padding
	Rect rect_padded=rect;
	rect_padded.x-=(padding_ratio-1)*(rect.width)/2.0;
	rect_padded.width=rect_padded.width*padding_ratio;
	rect_padded.y-=(padding_ratio-1)*(rect.height)/2.0;
	rect_padded.height=rect_padded.height*padding_ratio;
	rect=rect_padded;

	//crop box
	// Rect roi_rect_in_raw(rect);
	// Rect raw_rect_in_roi(0-rect.x, 0-rect.y, img.cols, img.rows);
	// Rect cropped_rect_in_raw=roi_rect_in_raw&Rect(0, 0, img.cols, img.rows);
	// Rect cropped_rect_in_roi=raw_rect_in_roi&Rect(0, 0, rect.width, rect.height);

	//crop
	// printf("Crop: \n");
	for(size_t i=0;i<lanmdmark.size();i++) {
		// if (i==0) {
		// 	fprintf(stderr, "%f\n", lanmdmark[i].x);
		// } 
		lanmdmark[i].x-=rect.x;
		lanmdmark[i].y-=rect.y;
		// if (i==0) {
		// 	fprintf(stderr, "%f\n", lanmdmark[i].x);
		// } 
		lanmdmark[i].x*=(double)(size_output.width)/(double)rect.width;
		lanmdmark[i].y*=(double)(size_output.height)/(double)rect.height;
		// if (i==0) {
		// 	fprintf(stderr, "%f\n", lanmdmark[i].x);
		// } 
	}
	// fprintf(stderr, "rect.width: %d size_output.width: %d\n", rect.width, size_output.width);
	// fprintf(stderr, "rect.height: %d size_output.height: %d\n", rect.height, size_output.height);
	// fprintf(stderr, "%d %d\n", rect.x, rect.y);
}

//inverse crop a landmark with padding ratio to raw image
void InvCropPointWithPadding(const cv::Size& img_size, Rect rect, float padding_ratio, Size size_output, vector<Point2f>& lanmdmark) {
	//padding
	Rect rect_padded=rect;
	rect_padded.x-=(padding_ratio-1)*(rect.width)/2.0;
	rect_padded.width=rect_padded.width*padding_ratio;
	rect_padded.y-=(padding_ratio-1)*(rect.height)/2.0;
	rect_padded.height=rect_padded.height*padding_ratio;
	rect=rect_padded;

	//crop box
	// Rect roi_rect_in_raw(rect);
	// Rect raw_rect_in_roi(0-rect.x, 0-rect.y, img_size.width, img_size.height);
	// Rect cropped_rect_in_raw=roi_rect_in_raw&Rect(0, 0, img_size.width, img_size.height);
	// Rect cropped_rect_in_roi=raw_rect_in_roi&Rect(0, 0, rect.width, rect.height);

	//inv crop
	// printf("Inv: \n");
	for(size_t i=0;i<lanmdmark.size();i++) {
		// if (i==0) {
		// 	fprintf(stderr, "%f\n", lanmdmark[i].x);
		// } 
		lanmdmark[i].x*=(float)(rect.width)/size_output.width;
		lanmdmark[i].y*=(float)(rect.height)/size_output.height;
		// if (i==0) {
		// 	fprintf(stderr, "%f\n", lanmdmark[i].x);
		// } 
		lanmdmark[i].x+=rect.x;
		lanmdmark[i].y+=rect.y;
		// if (i==0) {
		// 	fprintf(stderr, "%f\n", lanmdmark[i].x);
		// } 
	}
	// fprintf(stderr, "rect.width: %d size_output.width: %d\n", rect.width, size_output.width);
	// fprintf(stderr, "rect.height: %d size_output.height: %d\n", rect.height, size_output.height);
	// fprintf(stderr, "%d %d\n", rect.x, rect.y);
}

//inverse crop a landmark with padding ratio to raw image
void InvCropPointWithPadding(const Mat& img, Rect rect, float padding_ratio, Size size_output, vector<Point2f>& lanmdmark) {
	InvCropPointWithPadding(img.size(), rect, padding_ratio, size_output, lanmdmark);
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

std::vector<cv::Point2f> AffinePose(const cv::Mat_<float>& affine_mat, const std::vector<cv::Point2f>& pose) {
	vector<Point2f> affined_pose(pose.size());
	for(size_t i=0; i<affined_pose.size(); i++) {
		affined_pose[i].x=affine_mat(0, 0)*pose[i].x+affine_mat(0, 1)*pose[i].y+affine_mat(0, 2);
		affined_pose[i].y=affine_mat(1, 0)*pose[i].x+affine_mat(1, 1)*pose[i].y+affine_mat(1, 2);
	}
	return affined_pose;
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

std::vector<cv::Point2d> InvAffinePoseD(const cv::Mat_<double>& affine_mat, const std::vector<cv::Point2d>& pose) {
	vector<Point2d> inv_pose(pose.size());
	for(size_t i=0; i<inv_pose.size(); i++) {
		inv_pose[i].x=pose[i].x-affine_mat(0, 2);
		inv_pose[i].y=pose[i].y-affine_mat(1, 2);
	}
	double scale=affine_mat(0, 0)*affine_mat(0, 0)+affine_mat(0, 1)*affine_mat(0, 1);
	double inv_00=affine_mat(0, 0)/scale;
	double inv_01=-1*affine_mat(0, 1)/scale;
	double inv_10=-1*inv_01;
	double inv_11=inv_00;
	for(size_t i=0; i<inv_pose.size(); i++) {
		inv_pose[i]=Point2d(inv_00*inv_pose[i].x+inv_01*inv_pose[i].y,
				inv_10*inv_pose[i].x+inv_11*inv_pose[i].y);
	}
	return inv_pose;
}

vector<Point2f> LabelToLandmark(const vector<float> &label) {
	vector<Point2f> landmark(label.size()/2);
	for(size_t i=0;i<landmark.size();i++) {
		landmark[i].x=label[2*i];
		landmark[i].y=label[2*i+1];
	}

	return landmark;
}

vector<float> LandmarkToLabel(const vector<Point2f> &landmark) {
	vector<float> label(landmark.size()*2);
	for(size_t i=0;i<landmark.size();i++) {
		label[2*i]=landmark[i].x;
		label[2*i+1]=landmark[i].y;
	}

	return label;
}

vector<Rect2f> LabelToRect(const vector<float> &label) {
	vector<Rect2f> rect;
	for(size_t i=0;i<rect.size();i++) {
		if(label[5*i]<0) {
			continue;
		}
		Rect2f one_rect(label[5*i+1], label[5*i+2], label[5*i+3]-label[5*i+1], label[5*i+4]-label[5*i+2]);
		rect.push_back(one_rect);
	}

	return rect;
}

void SafeMatCrop(const Mat &image, Rect rect, Mat &cropped_image) {
	//crop box
	Rect roi_rect_in_raw(rect);
	Rect raw_rect_in_roi(0-rect.x, 0-rect.y, image.cols, image.rows);
	Rect cropped_rect_in_raw=roi_rect_in_raw&Rect(0, 0, image.cols, image.rows);
	Rect cropped_rect_in_roi=raw_rect_in_roi&Rect(0, 0, rect.width, rect.height);

	//crop
	cropped_image.create(rect.height, rect.width, image.type());
	cropped_image.setTo(Scalar(127, 127, 127, 0));
	Mat cropped_in_roi=cropped_image(cropped_rect_in_roi);
	Mat cropped_in_raw=image(cropped_rect_in_raw);
	cropped_in_raw.copyTo(cropped_in_roi);
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

float GetMeanValue(const std::vector<float>& data) {
	CHECK_GT(data.size(), 0);
	float total=0.0f;
	for(const auto& scalar: data) {
		total+=scalar;
	}
	return total/data.size();
}

float IoU(const cv::Rect& r1, const cv::Rect& r2) {
	float inter=(r1&r2).area();
	return inter/(r1.area()+r2.area()-inter);
}

float Distance(const cv::Point2f& p1, const cv::Point2f& p2) {
	return std::sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}


// definition:
// 0: right eye top
// 1: right eye bottom
// 2: left eye top
// 3: left eye bottom
// 4: right eyebrow
// 5: left eyebrow
// 6: nose contour
// 7: nose bridge
// 8: mouse top top
// 9: mouse top bottom
// 10 mouse bottom top
// 11: mouse bottom bottom
// 12: contour

// @Wenyan Wu 2017/03/01 
// change label.points[11] for all of the mouth boundary in one line
bool IsValid(const Label_2016& label) {
	if(label.points.size()!=13) {
		return false;
	}

	if(label.points[4].size()!=9 || label.points[5].size()!=9 ||
			label.points[6].size()<3 || label.points[7].size()<2 ||
			label.points[8].size()<2 || label.points[12].size()<3) {
		return false;
	}
	if(!(label.points[0].size()==0 && label.points[1].size()>=2) &&
			!(label.points[0].size()>=2 && label.points[1].size()==0) &&
			!(label.points[0].size()>=2 && label.points[1].size()>=2)) {
		return false;
	}
	if(!(label.points[2].size()==0 && label.points[3].size()>=2) &&
			!(label.points[2].size()>=2 && label.points[3].size()==0) &&
			!(label.points[2].size()>=2 && label.points[3].size()>=2)) {
		return false;
	}
	if(!(label.points[9].size()==0 && label.points[10].size()>=2 && label.points[11].size()>=2) &&
			!(label.points[9].size()>=2 && label.points[10].size()==0 && label.points[11].size()>=2) &&
			!(label.points[9].size()>=2 && label.points[10].size()>=2 && label.points[11].size()>=2) &&
			!(label.points[9].size()==0 && label.points[10].size()==0 && label.points[11].size()==0)) {
		return false;
	}
	return true;
}

// definition:
// 0: right eye top
// 1: right eye bottom
// 2: left eye top
// 3: left eye bottom
// 4: right eyebrow
// 5: left eyebrow
// 6: nose contour
// 7: nose bridge
// 8: mouse top top
// 9: mouse top bottom
// 10 mouse bottom top
// 11: mouse bottom bottom
// 12: contour
bool IsValidProfile(const Label_2016& label) {
	if(label.points.size()!=13) {
		return false;
	}
	if(label.points[6].size()<3 || label.points[7].size()<2 ||
		label.points[8].size()<2 || label.points[12].size()<3) {
		return false;
	}
	//for profile eyebrow
	if(!(label.points[4].size()==9 && label.points[5].size()==9) &&
		!(label.points[4].size()==0 && label.points[5].size()==9) &&
		!(label.points[4].size()==9 && label.points[5].size()==0)) {
		return false;
	}
	//for profile eyes
	if(label.points[0].size()<2 && label.points[1].size()<2 && label.points[2].size()<2 && label.points[3].size()<2) {
		return false;
	}
	//for close right eyes
	if(!(label.points[0].size()==0 && label.points[1].size()>=2) && 
		!(label.points[0].size()>=2 && label.points[1].size()==0) &&
		!(label.points[0].size()>=2 && label.points[1].size()>=2) &&
		!(label.points[0].size()==0 && label.points[1].size()==0)) {
		return false;
	}
	//for close left eyes
	if(!(label.points[2].size()==0 && label.points[3].size()>=2) &&
		!(label.points[2].size()>=2 && label.points[3].size()==0) &&
		!(label.points[2].size()>=2 && label.points[3].size()>=2) &&
		!(label.points[2].size()==0 && label.points[3].size()==0)) {
		return false;
	}
	//for close mouth
	if(!(label.points[9].size()==0 && label.points[10].size()>=2 && label.points[11].size()>=2) &&
		!(label.points[9].size()>=2 && label.points[10].size()==0 && label.points[11].size()>=2) &&
		!(label.points[9].size()>=2 && label.points[10].size()>=2 && label.points[11].size()>=2) &&
		!(label.points[9].size()==0 && label.points[10].size()==0 && label.points[11].size()==0)) {
		return false;
	}
	return true;
}

bool Is106ptValid(const std::vector<cv::Point2f>& landmarks) {
	static const float eps=0.05f;
	if(landmarks.size()!=106) {
		return false;
	}

	vector<Point2f> affine_src(2);
	affine_src[0]=landmarks[104];
	affine_src[1]=landmarks[105];
	vector<Point2f> affine_dst(2);
	affine_dst[0]=Point2f(0.0f, 0.0f);
	affine_dst[1]=Point2f(1.0f, 0.0f);
	Mat_<float> affine_mat=CalcAffineMatByPose(affine_src, affine_dst);
	vector<Point2f> affined_pose=AffinePose(affine_mat, landmarks);

	Point2f left_eyebrow_top=affined_pose[34]+affined_pose[35]+affined_pose[36]+affined_pose[37];
	Point2f left_eyebrow_bottom=affined_pose[64]+affined_pose[65]+affined_pose[66]+affined_pose[67];
	if(left_eyebrow_top.y-left_eyebrow_bottom.y>eps) {
		return false;
	}

	Point2f right_eyebrow_top=affined_pose[38]+affined_pose[39]+affined_pose[40]+affined_pose[41];
	Point2f right_eyebrow_bottom=affined_pose[68]+affined_pose[69]+affined_pose[70]+affined_pose[71];
	if(right_eyebrow_top.y-right_eyebrow_bottom.y>eps) {
		return false;
	}

	Point2f left_eye_top=affined_pose[53]+affined_pose[54]+affined_pose[72];
	Point2f left_eye_bottom=affined_pose[56]+affined_pose[57]+affined_pose[73];
	if(left_eye_top.y-left_eye_bottom.y>eps) {
		return false;
	}

	Point2f right_eye_top=affined_pose[59]+affined_pose[60]+affined_pose[75];
	Point2f right_eye_bottom=affined_pose[62]+affined_pose[63]+affined_pose[76];
	if(right_eye_top.y-right_eye_bottom.y>eps) {
		return false;
	}

	Point2f mouse_top_top=affined_pose[86]+affined_pose[87]+affined_pose[88];
	Point2f mouse_top_bottom=affined_pose[97]+affined_pose[98]+affined_pose[99];
	Point2f mouse_bottom_top=affined_pose[103]+affined_pose[102]+affined_pose[101];
	Point2f mouse_bottom_bottom=affined_pose[94]+affined_pose[93]+affined_pose[92];
	if(!(mouse_top_top.y-mouse_top_bottom.y<eps &&
			mouse_top_bottom.y-mouse_bottom_top.y<eps &&
			mouse_bottom_top.y-mouse_bottom_bottom.y<eps)) {
		return false;
	}
	return true;
}

bool Is106ptValidProfile(const std::vector<cv::Point2f>& landmarks) {
	static const float eps=0.05f;
	if(landmarks.size()!=106) {
		return false;
	}
	bool right_eye_is_invalid = false;
	bool left_eye_is_invalid = false;
	bool right_eyebrow_is_invalid = false;
	bool left_eyebrow_is_invalid = false;
	Point2f zero_point(0.0f, 0.0f);
	if (landmarks[33] == zero_point &&
		landmarks[34] == zero_point &&
		landmarks[64] == zero_point &&
		landmarks[35] == zero_point &&
		landmarks[65] == zero_point &&
		landmarks[36] == zero_point &&
		landmarks[66] == zero_point &&
		landmarks[37] == zero_point &&
		landmarks[67] == zero_point) {
		left_eyebrow_is_invalid = true;
	}
	if (landmarks[38] == zero_point &&
		landmarks[68] == zero_point &&
		landmarks[39] == zero_point &&
		landmarks[69] == zero_point &&
		landmarks[40] == zero_point &&
		landmarks[70] == zero_point &&
		landmarks[41] == zero_point &&
		landmarks[71] == zero_point &&
		landmarks[42] == zero_point) {
		right_eyebrow_is_invalid = true;
	}
	if (landmarks[52] == zero_point &&
		landmarks[53] == zero_point &&
		landmarks[72] == zero_point &&
		landmarks[54] == zero_point &&
		landmarks[55] == zero_point &&
		landmarks[57] == zero_point &&
		landmarks[73] == zero_point &&
		landmarks[56] == zero_point) {
		left_eye_is_invalid = true;
	}
	if (landmarks[58] == zero_point &&
		landmarks[59] == zero_point &&
		landmarks[75] == zero_point &&
		landmarks[60] == zero_point &&
		landmarks[61] == zero_point &&
		landmarks[63] == zero_point &&
		landmarks[76] == zero_point &&
		landmarks[62] == zero_point) {
		right_eye_is_invalid = true;
	}
	vector<Point2f> affine_src(2);
	if (left_eye_is_invalid || right_eye_is_invalid) {
		if (left_eye_is_invalid) {
			affine_src[0]=landmarks[43];
			affine_src[1]=landmarks[105];
		} else {
			affine_src[0]=landmarks[104];
			affine_src[1]=landmarks[43];
		}
	} else {
		affine_src[0]=landmarks[104];
		affine_src[1]=landmarks[105];
	}
	vector<Point2f> affine_dst(2);
	affine_dst[0]=Point2f(0.0f, 0.0f);
	affine_dst[1]=Point2f(1.0f, 0.0f);
	Mat_<float> affine_mat=CalcAffineMatByPose(affine_src, affine_dst);
	vector<Point2f> affined_pose=AffinePose(affine_mat, landmarks);

	Point2f left_eyebrow_top=affined_pose[34]+affined_pose[35]+affined_pose[36]+affined_pose[37];
	Point2f left_eyebrow_bottom=affined_pose[64]+affined_pose[65]+affined_pose[66]+affined_pose[67];
	if(!left_eyebrow_is_invalid && (left_eyebrow_top.y-left_eyebrow_bottom.y>eps)) {
		return false;
	}

	Point2f right_eyebrow_top=affined_pose[38]+affined_pose[39]+affined_pose[40]+affined_pose[41];
	Point2f right_eyebrow_bottom=affined_pose[68]+affined_pose[69]+affined_pose[70]+affined_pose[71];
	if(!right_eyebrow_is_invalid && (right_eyebrow_top.y-right_eyebrow_bottom.y>eps)) {
		return false;
	}

	Point2f left_eye_top=affined_pose[53]+affined_pose[54]+affined_pose[72];
	Point2f left_eye_bottom=affined_pose[56]+affined_pose[57]+affined_pose[73];
	if(!left_eye_is_invalid && (left_eye_top.y-left_eye_bottom.y>eps)) {
		return false;
	}

	Point2f right_eye_top=affined_pose[59]+affined_pose[60]+affined_pose[75];
	Point2f right_eye_bottom=affined_pose[62]+affined_pose[63]+affined_pose[76];
	if(!right_eye_is_invalid && (right_eye_top.y-right_eye_bottom.y>eps)) {
		return false;
	}

	Point2f mouse_top_top=affined_pose[86]+affined_pose[87]+affined_pose[88];
	Point2f mouse_top_bottom=affined_pose[97]+affined_pose[98]+affined_pose[99];
	Point2f mouse_bottom_top=affined_pose[103]+affined_pose[102]+affined_pose[101];
	Point2f mouse_bottom_bottom=affined_pose[94]+affined_pose[93]+affined_pose[92];
	if(!(mouse_top_top.y-mouse_top_bottom.y<eps &&
			mouse_top_bottom.y-mouse_bottom_top.y<eps &&
			mouse_bottom_top.y-mouse_bottom_bottom.y<eps)) {
		return false;
	}
	return true;
}

bool IsLeftToRight(const std::vector<cv::Point2f>& landmarks) {
	CHECK_GE(landmarks.size(), 2);
	return landmarks[0].x<(landmarks.end()-1)->x;
}

bool IsTopToBottom(const std::vector<cv::Point2f>& landmarks) {
	CHECK_GE(landmarks.size(), 2);
	return landmarks[0].y<(landmarks.end()-1)->y;
}

std::vector<cv::Point2f> KeepLeftToRight(const std::vector<cv::Point2f>& src, const cv::Mat_<float>& affine_mat) {
	if(IsLeftToRight(AffinePose(affine_mat, src))) {
		return src;
	}
	else {
		return Inverse(src);
	}
}

std::vector<cv::Point2f> KeepTopToBottom(const std::vector<cv::Point2f>& src, const cv::Mat_<float>& affine_mat) {
	if(IsTopToBottom(AffinePose(affine_mat, src))) {
		// printf("IsTopToBottom\n");
		return src;
	}
	else {
		// printf("IsNotTopToBottom\n");
		return Inverse(src);
	}
}


Label_2016 ToLabel2016(const std::vector<cv::Point2f>& label) {
	Label_2016 label_2016;
	label_2016.rotate=Label_2016::ROTATE_0;
	label_2016.points.resize(13);
	if(label.size()==106) {
		label_2016.points[0].push_back(label[58]);
		label_2016.points[0].push_back(label[59]);
		label_2016.points[0].push_back(label[75]);
		label_2016.points[0].push_back(label[60]);
		label_2016.points[0].push_back(label[61]);

		label_2016.points[1].push_back(label[58]);
		label_2016.points[1].push_back(label[63]);
		label_2016.points[1].push_back(label[76]);
		label_2016.points[1].push_back(label[62]);
		label_2016.points[1].push_back(label[61]);

		label_2016.points[2].push_back(label[52]);
		label_2016.points[2].push_back(label[53]);
		label_2016.points[2].push_back(label[72]);
		label_2016.points[2].push_back(label[54]);
		label_2016.points[2].push_back(label[55]);

		label_2016.points[3].push_back(label[52]);
		label_2016.points[3].push_back(label[57]);
		label_2016.points[3].push_back(label[73]);
		label_2016.points[3].push_back(label[56]);
		label_2016.points[3].push_back(label[55]);

		for(int i=38; i<43; i++) {
			label_2016.points[4].push_back(label[i]);
		}
		for(int i=68; i<72; i++) {
			label_2016.points[4].push_back(label[i]);
		}

		for(int i=33; i<38; i++) {
			label_2016.points[5].push_back(label[i]);
		}
		for(int i=64; i<68; i++) {
			label_2016.points[5].push_back(label[i]);
		}

		label_2016.points[6].push_back(label[80]);
		label_2016.points[6].push_back(label[82]);
		for(int i=47; i<52; i++) {
			label_2016.points[6].push_back(label[i]);
		}
		label_2016.points[6].push_back(label[83]);
		label_2016.points[6].push_back(label[81]);

		for(int i=43; i<47; i++) {
			label_2016.points[7].push_back(label[i]);
		}

		for(int i=84; i<91; i++) {
			label_2016.points[8].push_back(label[i]);
		}

		for(int i=96; i<101; i++) {
			label_2016.points[9].push_back(label[i]);
		}

		label_2016.points[10].push_back(label[96]);
		for(int i=103; i>99; i--) {
			label_2016.points[10].push_back(label[i]);
		}

		label_2016.points[11].push_back(label[84]);
		for(int i=95; i>89; i--) {
			label_2016.points[11].push_back(label[i]);
		}

		for(int i=0; i<33; i++) {
			label_2016.points[12].push_back(label[i]);
		}
	}
	else if(label.size()==91) {
		label_2016.points[0].push_back(label[42]);
		label_2016.points[0].push_back(label[43]);
		label_2016.points[0].push_back(label[59]);
		label_2016.points[0].push_back(label[44]);
		label_2016.points[0].push_back(label[45]);

		label_2016.points[1].push_back(label[42]);
		label_2016.points[1].push_back(label[47]);
		label_2016.points[1].push_back(label[60]);
		label_2016.points[1].push_back(label[46]);
		label_2016.points[1].push_back(label[45]);

		label_2016.points[2].push_back(label[36]);
		label_2016.points[2].push_back(label[37]);
		label_2016.points[2].push_back(label[56]);
		label_2016.points[2].push_back(label[38]);
		label_2016.points[2].push_back(label[39]);

		label_2016.points[3].push_back(label[36]);
		label_2016.points[3].push_back(label[41]);
		label_2016.points[3].push_back(label[57]);
		label_2016.points[3].push_back(label[40]);
		label_2016.points[3].push_back(label[39]);

		for(int i=22; i<27; i++) {
			label_2016.points[4].push_back(label[i]);
		}
		for(int i=52; i<56; i++) {
			label_2016.points[4].push_back(label[i]);
		}

		for(int i=17; i<22; i++) {
			label_2016.points[5].push_back(label[i]);
		}
		for(int i=48; i<52; i++) {
			label_2016.points[5].push_back(label[i]);
		}

		label_2016.points[6].push_back(label[64]);
		label_2016.points[6].push_back(label[66]);
		for(int i=31; i<36; i++) {
			label_2016.points[6].push_back(label[i]);
		}
		label_2016.points[6].push_back(label[67]);
		label_2016.points[6].push_back(label[65]);

		for(int i=27; i<31; i++) {
			label_2016.points[7].push_back(label[i]);
		}

		for(int i=71; i<78; i++) {
			label_2016.points[8].push_back(label[i]);
		}

		for(int i=83; i<88; i++) {
			label_2016.points[9].push_back(label[i]);
		}

		label_2016.points[10].push_back(label[83]);
		for(int i=90; i>87; i--) {
			label_2016.points[10].push_back(label[i]);
		}

		label_2016.points[11].push_back(label[71]);
		for(int i=82; i>76; i--) {
			label_2016.points[11].push_back(label[i]);
		}

		for(int i=0; i<17; i++) {
			label_2016.points[12].push_back(label[i]);
		}
	}
	else {
		CHECK(false);
	}
	return label_2016;
}

void randomParaGenerator(cv::Mat image, std::vector<cv::Mat > &augmented_img, std::vector<cv::Mat > &augmented_trans, int sample_num) {
	static const float rotation_min = -10.0f;
	static const float rotation_max = 10.0f;
	static const float trans_x_min = -20.0f;
	static const float trans_x_max = 20.0f;
	static const float trans_y_min = -20.0f;
	static const float trans_y_max = 20.0f;
	static const float zoom_min = 0.9f;
	static const float zoom_max = 1.1f;
	static const float mirror_min = 0.0f;
	static const float mirror_max = 0.2f;

	random_device rd;
	mt19937 generator(rd());
	uniform_real_distribution<float> dis_rotation(rotation_min, rotation_max);
	uniform_real_distribution<float> dis_trans_x(trans_x_min, trans_x_max);
	uniform_real_distribution<float> dis_trans_y(trans_y_min, trans_y_max);
	uniform_real_distribution<float> dis_zoom(zoom_min, zoom_max);
	uniform_real_distribution<float> dis_mirror(mirror_min, mirror_max);

	Point2f center(image.cols/2.0f-0.5f, image.rows/2.0f-0.5f);

	for(int i=0; i<sample_num; i++){
		float rotation_sample = dis_rotation(generator);
		float trans_x_sample = dis_trans_x(generator);
		float trans_y_sample = dis_trans_y(generator);
		float zoom_sample = dis_zoom(generator);

		augmented_trans[i] = getRotationMatrix2D(center, rotation_sample, 1.0f);
		augmented_trans[i].at<double>(0, 2) += trans_x_sample ;
		augmented_trans[i].at<double>(1, 2) += trans_y_sample ;
		for(size_t j = 0; j < 3; j++){
			augmented_trans[i].at<double>(0, j) *= zoom_sample;
			augmented_trans[i].at<double>(1, j) *= zoom_sample;
		}
		augmented_trans[i].at<double>(0, 2) += (1 - zoom_sample) * center.x;
		augmented_trans[i].at<double>(1, 2) += (1 - zoom_sample) * center.y;

		// mirror about x axis in cropped image
		// deprecated
		// float mirror_sample = dis_mirror(generator);
		// if (mirror_sample >= 0.5f) {
		// 	augmented_trans[i].at<double>(0, 0) = -augmented_trans[i].at<double>(0, 0);
		// 	augmented_trans[i].at<double>(0, 1) = -augmented_trans[i].at<double>(0, 1);
		// 	augmented_trans[i].at<double>(0, 2) = output_img_width-augmented_trans[i].at<double>(0, 2);
		// }

		augmented_trans[i].convertTo(augmented_trans[i], CV_32F);
		warpAffine(image, augmented_img[i], augmented_trans[i], image.size(),
				INTER_LINEAR, BORDER_CONSTANT, Scalar(127, 127, 127, 0));
	}
}

bool sort_min (float i, float j) { 
	return (i < j); 
}

} // namespace alignment_tools
