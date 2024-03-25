#include <opencv2/opencv.hpp>
#include "inference.hpp"
#include "spdlog/spdlog.h"


int main(int argc, char *argv[]){
    float confThreshold = 0.1f;
    float iouThreshold = 0.1f;
    float maskThreshold = 0.1f;
    bool isGPU = false;

    spdlog::info("Start inference");

    std::string modelPath = "/doc/work/data/RIWO/data/calyx/runs/segment/train/weights/best.onnx";
    std::string imagePath = "/doc/work/data/RIWO/data/calyx/images/20210914_105512764176_rgb_trigger003_apple1.png";

    ONNXInf inf = ONNXInf(modelPath, isGPU, confThreshold, iouThreshold, maskThreshold);
 
    cv::Mat image = cv::imread(imagePath);
    std::vector<Detection> result = inf.predict(image);

    for (auto &det : result){
        spdlog::info("Id: {}, Accu: {}, Bbox: ({}, {}, {}, {})", det.id, det.accu, det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);
    }

    spdlog::info("Inference done");

    // utils::visualizeDetection(image, result, classNames);


    return 0;
}