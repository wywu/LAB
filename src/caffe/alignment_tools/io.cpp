#include <string>
#include <vector>
#include <cstdio>
#include <cmath>

#include <glog/logging.h>

#include "caffe/alignment_tools/io.hpp"
#include "caffe/alignment_tools/util.hpp"

using namespace std;

namespace alignment_tools {

FILE* OpenFileOrStd(const std::string& file_name, const std::string& type) {
	CHECK(type=="w" || type=="r");
	if(file_name!="") {
		FILE* fp=fopen(file_name.c_str(), type.c_str());
		CHECK(fp!=NULL) << "Open failed: " << file_name;
		return fp;
	}
	else {
		return type=="w"?stdout:stdin;
	}
}

void CloseFileOrStd(FILE*& fp) {
	if(fp!=NULL && fp!=stdout && fp!=stdin) {
		fclose(fp);
	}
}

void ReadImageLabelList(const std::string& file_name, const int& label_num,
		std::vector<std::string>& image_list, std::vector<std::vector<float>>& label_list) {
	image_list.clear();
	label_list.clear();
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		vector<float> label;
		for(size_t i=0; i<label_num; i++) {
			float label_item;
			if(fscanf(fp, "%f", &(label_item))!=1) {
				break;
			}
			label.push_back(label_item);
		}
		if(label.size()!=label_num) {
			break;
		}
		fgetc(fp);
		char buf[max_path]="";
		fgets(buf, max_path, fp);
		string image(buf);
		image=image.substr(0, image.find_first_of("\n\r"));
		for(size_t i=0; i<image.size(); i++) {
			image[i]=(image[i]=='\\')?'/':image[i];
		}
		if(image.size()==0) {
			break;
		}
		label_list.push_back(label);
		image_list.push_back(image);
	}
	CloseFileOrStd(fp);
	if(label_list.size()!=image_list.size() || image_list.size()==0) {
		label_list.clear();
		image_list.clear();
	}
}

std::vector<std::vector<float>> ReadLabelList(const std::string& file_name, const int& label_num) {
	vector<vector<float>> label_list;
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		vector<float> label;
		for(size_t i=0; i<label_num; i++) {
			float label_item;
			if(fscanf(fp, "%f", &(label_item))!=1) {
				break;
			}
			label.push_back(label_item);
		}
		if(label.size()!=label_num) {
			break;
		}
		label_list.push_back(label);
	}
	CloseFileOrStd(fp);
	return label_list;
}

void WriteImageLabelList(const std::string& file_name,
		const std::vector<std::string>& image_list,const std::vector<std::vector<float>>& label_list) {
	CHECK_EQ(image_list.size(), label_list.size());
	FILE* fp=OpenFileOrStd(file_name.c_str(), "w");
	for(size_t i=0; i<image_list.size(); i++) {
		for(size_t j=0; j<label_list[i].size(); j++) {
			fprintf(fp, "%f ", label_list[i][j]);
		}
		fprintf(fp, "%s\n", image_list[i].c_str());
	}
	CloseFileOrStd(fp);
}



} // namespace alignment_tools