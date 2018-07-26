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

void ReadImageLabelListD(const std::string& file_name, const int& label_num,
		std::vector<std::string>& image_list, std::vector<std::vector<double>>& label_list) {
	image_list.clear();
	label_list.clear();
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		vector<double> label;
		for(size_t i=0; i<label_num; i++) {
			double label_item;
			if(fscanf(fp, "%lf", &(label_item))!=1) {
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

void WriteImageLabelListD(const std::string& file_name,
		const std::vector<std::string>& image_list,const std::vector<std::vector<double>>& label_list) {
	CHECK_EQ(image_list.size(), label_list.size());
	FILE* fp=OpenFileOrStd(file_name.c_str(), "w");
	for(size_t i=0; i<image_list.size(); i++) {
		for(size_t j=0; j<label_list[i].size(); j++) {
			fprintf(fp, "%lf ", label_list[i][j]);
		}
		fprintf(fp, "%s\n", image_list[i].c_str());
	}
	CloseFileOrStd(fp);
}

std::vector<std::string> ReadImageList(const std::string& file_name) {
	vector<string> image_list;
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		char buf[max_line]="";
		fgets(buf, max_line, fp);
		string image(buf);
		image=image.substr(0, image.find_first_of("\n\r"));
		for(size_t i=0;i<image.size();i++) {
			image[i]=(image[i]=='\\')?'/':image[i];
		}
		if(image.size()==0) {
			break;
		}
		image_list.push_back(image);
	}
	CloseFileOrStd(fp);
	return image_list;
}

void WriteImageList(const std::string& file_name, const std::vector<std::string>& image_list) {
	FILE* fp=OpenFileOrStd(file_name.c_str(), "w");
	for(size_t i=0; i<image_list.size(); i++) {
		fprintf(fp, "%s\n", image_list[i].c_str());
	}
	CloseFileOrStd(fp);
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

std::vector<std::vector<double>> ReadLabelListD(const std::string& file_name, const int& label_num) {
	vector<vector<double>> label_list;
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		vector<double> label;
		for(size_t i=0; i<label_num; i++) {
			double label_item;
			if(fscanf(fp, "%lf", &(label_item))!=1) {
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

std::vector<int> ReadIndexList(const std::string& file_name) {
	vector<int> index_list;
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		int index_item;
		if(fscanf(fp, "%d", &(index_item))!=1) {
			break;
		}
		index_list.push_back(index_item);
	}
	CloseFileOrStd(fp);
	return index_list;
}

void WriteLabelList(const std::string& file_name, const std::vector<std::vector<float>>& label_list) {
	FILE* fp=OpenFileOrStd(file_name.c_str(), "w");
	for(size_t i=0; i<label_list.size(); i++) {
		for(size_t j=0; j<label_list[i].size(); j++) {
			fprintf(fp, "%f ", label_list[i][j]);
		}
		fprintf(fp, "\n");
	}
	CloseFileOrStd(fp);
}

void WriteLabelListD(const std::string& file_name, const std::vector<std::vector<double>>& label_list) {
	FILE* fp=OpenFileOrStd(file_name.c_str(), "w");
	for(size_t i=0; i<label_list.size(); i++) {
		for(size_t j=0; j<label_list[i].size(); j++) {
			fprintf(fp, "%lf ", label_list[i][j]);
		}
		fprintf(fp, "\n");
	}
	CloseFileOrStd(fp);
}

std::vector<float> StringToLabels(const std::string& src) {
	vector<float> labels;
	vector<bool> isnum(src.size()+1, false);
	for(size_t i=0;i<isnum.size();i++) {
		isnum[i]=src[i]=='.' || src[i]=='+' || src[i]=='-' || (src[i]>='0' && src[i]<='9');
	}
	for(size_t curser=0; curser<isnum.size(); curser++) {
		if(isnum[curser]) {
			size_t start=curser;
			for(curser++; curser<isnum.size(); curser++) {
				if(!isnum[curser]) {
					labels.push_back(atof(src.substr(start, curser-start).c_str()));
					break;
				}
			}
		}
	}
	return labels;
}

void ReadImageLabel2016List(const std::string& file_name,
		std::vector<std::string>& image_list, std::vector<Label_2016>& label_list) {
	image_list.clear();
	label_list.clear();
	FILE* fp=OpenFileOrStd(file_name.c_str(), "r");
	for(;;) {
		bool valid=true;
		char image_buf[max_path]="";
		int rotate;
		Label_2016 label;
		if(fscanf(fp, "%s%d", image_buf, &rotate)!=2) {
			break;
		}
		if(rotate==0) {
			label.rotate=Label_2016::ROTATE_0;
		}
		else if(rotate==90) {
			label.rotate=Label_2016::ROTATE_90;
		}
		else if(rotate==180) {
			label.rotate=Label_2016::ROTATE_180;
		}
		else if(rotate==270) {
			label.rotate=Label_2016::ROTATE_270;
		}
		else {
			valid=false;
		}
		char line_buf[max_line]="";
		fgets(line_buf, max_line, fp);
		for(int i=0; i<13; i++) {
			char buf[max_line]="";
			fgets(buf, max_line, fp);
			vector<float> line_float=StringToLabels(string(buf));
			CHECK_GE(line_float.size(), 1);
			if(line_float[0]<0.5f) {
				line_float.resize(1);
			}
			vector<float> one_label;
			one_label.insert(one_label.end(), line_float.begin()+1, line_float.end());
			if(one_label.size()%2==0) {
				label.points.push_back(ToPoints(one_label));
			}
			else {
				valid=false;
			}
		}
		if(valid) {
			image_list.push_back(string(image_buf));
			label_list.push_back(label);
		}
	}
	CloseFileOrStd(fp);
}

void WriteImageLabel2016List(const std::string& file_name,
		const std::vector<std::string>& image_list, const std::vector<Label_2016>& label_list) {
	CHECK_EQ(image_list.size(), label_list.size());
	FILE* fp=OpenFileOrStd(file_name.c_str(), "w");
	for(size_t i=0; i<image_list.size(); i++) {
		CHECK_EQ(label_list[i].points.size(), label_list[i].points.size());
		CHECK_EQ(label_list[i].points.size(), 13);
		fprintf(fp, "%s %d\n", image_list[i].c_str(), label_list[i].rotate);
		for(size_t j=0; j<label_list[i].points.size(); j++) {
			fprintf(fp, "%d ", label_list[i].points[j].size()>0?1:0);
			for(size_t k=0; k<label_list[i].points[j].size(); k++) {
				fprintf(fp, "%f %f ", label_list[i].points[j][k].x, label_list[i].points[j][k].y);
			}
			fprintf(fp, "\n");
		}
	}
	CloseFileOrStd(fp);
}

} // namespace alignment_tools
