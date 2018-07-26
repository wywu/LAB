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
void WriteImageLabelList(const std::string& file_name,
		const std::vector<std::string>& image_list, const std::vector<std::vector<float>>& label_list);
std::vector<std::vector<float>> ReadLabelList(const std::string& file_name, const int& label_num);

} // namespace alignment_tools

#endif // #ifdef __IO_HPP__
