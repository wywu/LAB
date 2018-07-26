#ifndef __IO_HPP__
#define __IO_HPP__

#include <string>
#include <vector>
#include <opencv2/core/core.hpp>

namespace alignment_tools {

static const size_t max_path=1000;
static const size_t max_line=10000;

FILE* OpenFileOrStd(const std::string& file_name, const std::string& type);
void CloseFileOrStd(FILE*& fp);

void ReadImageLabelList(const std::string& file_name, const int& label_num,
		std::vector<std::string>& image_list, std::vector<std::vector<float>>& label_list);
void ReadImageLabelListD(const std::string& file_name, const int& label_num,
		std::vector<std::string>& image_list, std::vector<std::vector<double>>& label_list);
void WriteImageLabelList(const std::string& file_name,
		const std::vector<std::string>& image_list, const std::vector<std::vector<float>>& label_list);
void WriteImageLabelListD(const std::string& file_name,
		const std::vector<std::string>& image_list,const std::vector<std::vector<double>>& label_list);
std::vector<std::string> ReadImageList(const std::string& file_name);
void WriteImageList(const std::string& file_name, const std::vector<std::string>& image_list);
std::vector<std::vector<float>> ReadLabelList(const std::string& file_name, const int& label_num);
std::vector<std::vector<double>> ReadLabelListD(const std::string& file_name, const int& label_num);
void WriteLabelList(const std::string& file_name, const std::vector<std::vector<float>>& label_list);
void WriteLabelListD(const std::string& file_name, const std::vector<std::vector<double>>& label_list);
std::vector<int> ReadIndexList(const std::string& file_name);

std::vector<float> StringToLabels(const std::string& src);

class Label_2016 {
public:
	enum {
		ROTATE_0=0,
		ROTATE_90=90,
		ROTATE_180=180,
		ROTATE_270=270
	} rotate;
	std::vector<std::vector<cv::Point2f>> points;
};

void ReadImageLabel2016List(const std::string& file_name,
		std::vector<std::string>& image_list, std::vector<Label_2016>& label_list);
void WriteImageLabel2016List(const std::string& file_name,
		const std::vector<std::string>& image_list, const std::vector<Label_2016>& label_list);

} // namespace alignment_tools

#endif // #ifdef __IO_HPP__
