#include <opencv2/opencv.hpp>
#include "inference.hpp"
#include "spdlog/spdlog.h"

int main(int argc, char *argv[]){
    float confThreshold = 0.4f;
    float iouThreshold = 0.4f;
    float maskThreshold = 0.5f;
    bool isGPU = false;

    spdlog::info("Start inference");

    // std::string modelPath = "/doc/work/data/RIWO/data/calyx/runs/segment/train/weights/best.onnx";
    // std::string imagePath = "/doc/work/data/RIWO/data/calyx/images/20210914_105512764176_rgb_trigger003_apple1.png";

    // ONNXInf inf = ONNXInf(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);

    // cv::Mat image = cv::imread(imagePath);
    // std::vector<Yolov8Result> result = inf.predict(image);

    // utils::visualizeDetection(image, result, classNames);

    return 0;
}