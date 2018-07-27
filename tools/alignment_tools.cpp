#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include <time.h>
#include <mutex>
#include <unordered_set>
#include <omp.h>
#include <algorithm>
#include <queue>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <caffe/caffe.hpp>
// #include <st_face/cv_face.h>

#include "caffe/alignment_tools/util.hpp"
#include "caffe/alignment_tools/io.hpp"

#include "caffe/common.hpp"

using namespace std;
using namespace cv;
using namespace caffe;
using namespace alignment_tools;


DEFINE_int32(thread_num, 1, "thread_num");
DEFINE_string(input_file_1, "", "input_file_1");
DEFINE_string(input_file_2, "", "input_file_2");
DEFINE_string(output_file_1, "", "output_file_1");
DEFINE_string(output_file_2, "", "output_file_2");
DEFINE_string(output_file_3, "", "output_file_3");
DEFINE_string(output_file_4, "", "output_file_4");
DEFINE_string(input_folder, "", "input_folder");
DEFINE_string(output_folder, "", "output_folder");
DEFINE_string(output_prefix, "", "output_prefix");
DEFINE_int32(label_num, 0, "label_num");
DEFINE_int32(index_num, 0, "index_num");
DEFINE_int32(output_width, 0, "output_width");
DEFINE_int32(output_height, 0, "output_height");
DEFINE_double(padding_ratio, 0.0, "padding_ratio");
DEFINE_int32(min_bbox_size, 0, "min_bbox_size");
DEFINE_int32(max_bbox_size, 0, "max_bbox_size");
DEFINE_int32(bbox_size_level, 0, "bbox_size_level");
DEFINE_double(threshold, 1.0, "threshold");
DEFINE_int32(select_num, 0, "select_num");
DEFINE_string(command_line, "", "command_line");
DEFINE_double(ratio, 0.0, "ratio");
DEFINE_string(model_path, "", "model_path");
DEFINE_int32(gpu_id, -1, "gpu_id");
DEFINE_int32(output_size, 0, "output_size");
DEFINE_int32(key_index, 0, "key_index");
DEFINE_int32(start_index, 0, "start_index");
DEFINE_int32(end_index, 0, "end_index");
DEFINE_int32(decrease, 0, "decrease");
DEFINE_int32(iteration_num, 0, "iteration_num");
DEFINE_int32(list_index, 0, "list_index");
DEFINE_int32(model_type, 0, "model_type");
DEFINE_int32(test_type, 0, "test_type");
DEFINE_double(rotation_angle, 0.0, "rotation_angle");


typedef int (*command_function_type) ();
vector<string> command_string;
vector<command_function_type> command_function;
#define REG_COMMAND(name, func) \
  command_string.push_back(name); \
  command_function.push_back(func);

int Help(void);
int RunTestOnWFLW(void);

int main(int argc, char **argv) {
#ifdef USE_MPI
  MPI_Init(&argc, &argv);
#endif

REG_COMMAND("help", Help);
REG_COMMAND("run_test_on_wflw", RunTestOnWFLW);

#ifndef GFLAGS_GFLAGS_H_
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  google::LogToStderr();
#else
  gflags::ParseCommandLineFlags(&argc, &argv, true);
#endif  // GFLAGS_GFLAGS_H_

  string command("help");
  if(argc>1) {
    command=string(argv[1]);
  }
  for(size_t i=0; i<command_string.size(); i++) {
    if(command==command_string[i]) {
      return (*command_function[i])();
    }
  }
  return Help();

#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}

int Help(void) {

  fprintf(stderr, "./alignment_tools command [-Parameter] [Value] (or [--Parameter=Value])\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Available Parameter: \n");
  fprintf(stderr, "\tthread_num\n");
  fprintf(stderr, "\tinput_file_n\n");
  fprintf(stderr, "\toutput_file_n\n");
  fprintf(stderr, "\tinput_folder\n");
  fprintf(stderr, "\toutput_folder\n");
  fprintf(stderr, "\toutput_prefix\n");
  fprintf(stderr, "\tlabel_num\n");
  fprintf(stderr, "\toutput_width\n");
  fprintf(stderr, "\toutput_height\n");
  fprintf(stderr, "\tpadding_ratio\n");
  fprintf(stderr, "\tmin_bbox_size\n");
  fprintf(stderr, "\tmax_bbox_size\n");
  fprintf(stderr, "\tbbox_size_level\n");
  fprintf(stderr, "\tthreshold\n");
  fprintf(stderr, "\tselect_num\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Available commmand: \n");
  for(size_t i=0; i<command_string.size(); i++) {
    fprintf(stderr, "\t%s\n", command_string[i].c_str());
  }

  return 0;
}


int RunTestOnWFLW(void) {

  CHECK_GT(FLAGS_label_num, 0);
  CHECK_EQ(FLAGS_label_num%2, 0);
  CHECK_GT(FLAGS_thread_num, 0);

  fprintf(stderr, "Input list file: %s\n", FLAGS_input_file_1==""?"stdin":FLAGS_input_file_1.c_str());
  fprintf(stderr, "Input mean pose file: %s\n", FLAGS_input_file_2.c_str());
  fprintf(stderr, "Input image folder: %s\n", FLAGS_input_folder.c_str());
  fprintf(stderr, "Output list file: %s\n", FLAGS_output_file_1==""?"stdout":FLAGS_output_file_1.c_str());
  fprintf(stderr, "Model path: %s\n", FLAGS_model_path.c_str());
  fprintf(stderr, "Number of label: %d\n", FLAGS_label_num);
  fprintf(stderr, "Thread Num: %d\n", FLAGS_thread_num);
  int instance_num;
  fprintf(stderr, "Use CPU.\n");
  fprintf(stderr, "Thread num: %d\n", FLAGS_thread_num);
  caffe::Caffe::set_mode(caffe::Caffe::CPU);
  instance_num=FLAGS_thread_num;

  const int meanpose_num = 71*2;
  const int img_size = 384;
  const int crop_size = 256;
  const float zoom = 1.0;

  vector<string> image_list;
  vector<vector<float>> label_list;
  ReadImageLabelList(FLAGS_input_file_1, FLAGS_label_num, image_list, label_list);
  CHECK_GT(image_list.size(), 0);
  CHECK_EQ(image_list.size(), label_list.size());
  fprintf(stderr, "Total %zd items.\n", image_list.size());

  vector<Point2f> mean_pose;
  vector<vector<float> > temp_list=ReadLabelList(FLAGS_input_file_2, meanpose_num);
  CHECK_EQ(temp_list.size(), 1);
  mean_pose=ToPoints(temp_list[0]);

  vector<caffe::shared_ptr<Net<float>>> deep_align(instance_num);
  vector<caffe::shared_ptr<Blob<float>>> input_blob(instance_num);
  vector<caffe::shared_ptr<Blob<float>>> output_blob(instance_num);
  for(size_t i=0; i<instance_num; i++) {
    deep_align[i].reset(new Net<float>(FLAGS_model_path+"rel.prototxt", TEST));
    deep_align[i]->CopyTrainedLayersFrom(FLAGS_model_path+"model.bin");
    input_blob[i]=deep_align[i]->blob_by_name("data");
    output_blob[i]=deep_align[i]->blob_by_name("result");
  }
  const int input_channel=input_blob[0]->channels();
  const int input_height=input_blob[0]->height();
  const int input_width=input_blob[0]->width();
  CHECK(input_channel==1 || input_channel==3);
  CHECK_EQ(input_width,crop_size);
  CHECK_EQ(input_height,crop_size);

  //crop affine mat calculate
  Point2f center;
  center.x = img_size/2.0 - 0.5;
  center.y = img_size/2.0 - 0.5;
  Mat affine_mat_crop =  getRotationMatrix2D(center, 0, 1);
  for(int i = 0; i < 3; i++){
    affine_mat_crop.at<double>(0, i) *= zoom;
    affine_mat_crop.at<double>(1, i) *= zoom;
  }
  affine_mat_crop.at<double>(0, 2) += (1 - zoom) * center.x;
  affine_mat_crop.at<double>(1, 2) += (1 - zoom) * center.y;
  affine_mat_crop.at<double>(0, 2) -= (img_size-crop_size)/2.0;
  affine_mat_crop.at<double>(1, 2) -= (img_size-crop_size)/2.0;


  vector<vector<float>> predict_list(image_list.size());
  vector<bool> image_list_valid(image_list.size(), true);
  SafeCounter counter(image_list.size(), 10);
#pragma omp parallel for num_threads(FLAGS_thread_num) schedule(static)
  for(size_t i=0;i<image_list.size();i++) {
    counter++;
    int thread_id=omp_get_thread_num();

    Mat image=imread(FLAGS_input_folder+image_list[i]);
    if(image.data==NULL) {
      fprintf(stderr, "Warning: Open failed: %s\n", (FLAGS_input_folder+image_list[i]).c_str());
      image_list_valid[i]=false;
      continue;
    }
    if(input_channel==1) {
      ConvertImageToGray(image);
    } else {
      ConvertImageToBGR(image);
    }

    vector<float> label_71pt_list(71*2);
    for (size_t j=0; j<76; j++) {
      label_71pt_list[j] = label_list[i][j];
    }
    for (size_t j=76; j<86; j++) {
      label_71pt_list[j] = label_list[i][j+8];
    }
    for (size_t j=86; j<94; j++) {
      label_71pt_list[j] = label_list[i][j+16];
    }
    label_71pt_list[94] = label_list[i][120];
    label_71pt_list[95] = label_list[i][121];
    label_71pt_list[96] = label_list[i][128];
    label_71pt_list[97] = label_list[i][129];
    label_71pt_list[98] = label_list[i][136];
    label_71pt_list[99] = label_list[i][137];
    label_71pt_list[100] = label_list[i][144];
    label_71pt_list[101] = label_list[i][145];
    for (size_t j=102; j<142; j++) {
      label_71pt_list[j] = label_list[i][j+50];
    }
    vector<Point2f> landmark=ToPoints(label_71pt_list);

    Mat cropped_face_0;
    Mat cropped_face_1;
    Mat affine_mat=CalcAffineMatByPose(landmark, mean_pose);

    float a = affine_mat.at<float>(0,0);
    float b = affine_mat.at<float>(0,1);
    float tx = affine_mat.at<float>(0,2);
    float c = affine_mat.at<float>(1,0);
    float d = affine_mat.at<float>(1,1);
    float ty = affine_mat.at<float>(1,2);
    float x_trans = tx;
    float y_trans = ty;
    float x_scale = (a>0) ? sqrt(a*a + b*b) : -sqrt(a*a + b*b);
    float y_scale = (d>0) ? sqrt(c*c + d*d) : -sqrt(c*c + d*d);
    float cos_ = a / x_scale;
    float sin_ = c / y_scale;
    float width_half = img_size / 2;
    float height_half = img_size / 2;

    Mat_<float> transform_matrix(2, 3);
    transform_matrix(0,0) = a*cos_ + c*sin_;
    transform_matrix(0,1) = b*cos_ + d*sin_;
    transform_matrix(0,2) = x_trans*cos_ + y_trans*sin_ - width_half*cos_ - width_half*sin_ + width_half;
    transform_matrix(1,0) = -a*sin_ + c*cos_;
    transform_matrix(1,1) = -b*sin_ + d*cos_;
    transform_matrix(1,2) = -x_trans*sin_ + y_trans*cos_ + height_half*sin_ - height_half*cos_ + height_half;

    warpAffine(image, cropped_face_0, transform_matrix, Size(img_size, img_size),
      INTER_LINEAR, BORDER_CONSTANT, Scalar(127, 127, 127, 0));
    warpAffine(cropped_face_0, cropped_face_1, affine_mat_crop, Size(crop_size, crop_size),
      INTER_LINEAR, BORDER_CONSTANT, Scalar(127, 127, 127, 0));

    cropped_face_1.convertTo(cropped_face_1, CV_32F);
    NormalizeImage(cropped_face_1);
    Copy(cropped_face_1, *(input_blob[thread_id]));
    deep_align[thread_id]->ForwardPrefilled();
    vector<Point2f> landmark_98pt=ToPoints(ToVector(*(output_blob[thread_id])));

    // plot
    // Mat show = cropped_face_1;
    // DrawPointsOnImage(show, ToLabel(landmark_98pt));
    // imshow("tracking", show);
    // fprintf(stderr, "show\n");
    // waitKey(0);


    landmark_98pt=InvAffinePose(affine_mat_crop, landmark_98pt);
    landmark_98pt=InvAffinePose(transform_matrix, landmark_98pt);
    predict_list[i]=ToLabel(landmark_98pt);
  }

  WriteImageLabelList(FLAGS_output_file_1, Filter(image_list, image_list_valid),
      Filter(predict_list, image_list_valid));
  return 0;
}