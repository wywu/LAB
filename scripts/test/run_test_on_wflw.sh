#!/usr/bin/env bash


##  parameter description
    #  --input_file_1: ground truth list of 98 landmarks
    #  --input_file_2: precomputed meanpose
    #  --input_folder: path of testing images
    #  --model_path: path of pretrained model
    #  --output_file_1: predicted list of 98 landmarks
    #  --label_num: 2 * num of landmarks
    #  --thread_num: number of threads

MODEL=$1
mkdir -p ./evaluation/WFLW/WFLW_${MODEL}_result

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test_largepose.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_largepose.txt --label_num=196 --thread_num=12
echo "list 1 done"

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test_expression.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_expression.txt --label_num=196 --thread_num=12
echo "list 2 done"

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test_illumination.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_illumination.txt --label_num=196 --thread_num=12
echo "list 3 done"

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test_makeup.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_makeup.txt --label_num=196 --thread_num=1
echo "list 4 done"

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test_occlusion.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_occlusion.txt --label_num=196 --thread_num=12
echo "list 5 done"

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test_blur.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_blur.txt --label_num=196 --thread_num=12
echo "list 6 done"

./build/tools/alignment_tools run_test_on_wflw --input_file_1=./datasets/WFLW/WFLW_annotations/list_98pt_test.txt --input_file_2=./meanpose/meanpose_71pt.txt --input_folder=./datasets/WFLW/WFLW_images/ --model_path=./models/WFLW/WFLW_${MODEL}/ --output_file_1=./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_test.txt --label_num=196 --thread_num=12
echo "list 7 done"

cat ./datasets/WFLW/WFLW_annotations/list_98pt_test_largepose.txt ./datasets/WFLW/WFLW_annotations/list_98pt_test_expression.txt ./datasets/WFLW/WFLW_annotations/list_98pt_test_illumination.txt ./datasets/WFLW/WFLW_annotations/list_98pt_test_makeup.txt ./datasets/WFLW/WFLW_annotations/list_98pt_test_occlusion.txt ./datasets/WFLW/WFLW_annotations/list_98pt_test_blur.txt ./datasets/WFLW/WFLW_annotations/list_98pt_test.txt > ./evaluation/WFLW/WFLW_${MODEL}_result/gt_release.txt

cat ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_largepose.txt ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_expression.txt ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_illumination.txt ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_makeup.txt ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_occlusion.txt ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_blur.txt ./evaluation/WFLW/WFLW_${MODEL}_result/pred_98pt_test.txt > ./evaluation/WFLW/WFLW_${MODEL}_result/pred_release.txt

