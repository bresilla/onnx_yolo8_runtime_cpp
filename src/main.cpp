    // std::string model_path = "/doc/work/data/RIWO/data/calyx/runs/segment/train/weights/best.onnx";

#include <regex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <ctime>
#include "inference.hpp"

#include "spdlog/spdlog.h"

int main(int argc, char *argv[])
{
    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold = 0.5f;
    bool isGPU = true;

    std::string modelPath = "/doc/work/data/RIWO/data/calyx/runs/segment/train/weights/best.onnx";
    std::string imagePath = "/doc/work/data/RIWO/data/calyx/images/20210914_105512764176_rgb_trigger003_apple1.png";


    YOLOPredictor predictor{nullptr};
    try {
        predictor = YOLOPredictor(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);
        std::cout << "Model was initialized." << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    
    cv::Mat image = cv::imread(imagePath);
    std::vector<Yolov8Result> result = predictor.predict(image);
    // utils::visualizeDetection(image, result, classNames);

    return 0;
}