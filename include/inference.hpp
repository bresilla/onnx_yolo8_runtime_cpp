#pragma once
#include <opencv2/opencv.hpp>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <utility>
#include <codecvt>
#include <fstream>

struct Yolov8Result {
    cv::Rect box;
    cv::Mat boxMask;
    float conf{};
    int classId{};
};

namespace utils{
    static std::vector<cv::Scalar> colors;

    size_t vectorProduct(const std::vector<int64_t> &vector);
    std::wstring charToWstring(const char *str);
    std::vector<std::string> loadNames(const std::string &path);
    void visualizeDetection(cv::Mat &image, std::vector<Yolov8Result> &results, const std::vector<std::string> &classNames);
    void letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape, const cv::Scalar &color, bool auto_, bool scaleFill, bool scaleUp, int stride);
    void scaleCoords(cv::Rect &coords, cv::Mat &mask, const float maskThreshold, const cv::Size &imageShape, const cv::Size &imageOriginalShape);

    template <typename T>
    T clip(const T &n, const T &lower, const T &upper);
}

class YOLOPredictor {
    public:
        explicit YOLOPredictor(std::nullptr_t){};
        YOLOPredictor(const std::string &modelPath, const bool &isGPU, float confThreshold, float iouThreshold, float maskThreshold);
        std::vector<Yolov8Result> predict(cv::Mat &image);
        int classNums = 80;

    private:
        Ort::Env env{nullptr};
        Ort::SessionOptions sessionOptions{nullptr};
        Ort::Session session{nullptr};

        void preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape);
        std::vector<Yolov8Result> postprocessing(const cv::Size &resizedImageShape, const cv::Size &originalImageShape, std::vector<Ort::Value> &outputTensors);
        static void getBestClassInfo(std::vector<float>::iterator it, float &bestConf, int &bestClassId, const int _classNums);
        cv::Mat getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos);
        bool isDynamicInputShape{};

        std::vector<const char *> inputNames;
        std::vector<Ort::AllocatedStringPtr> input_names_ptr;

        std::vector<const char *> outputNames;
        std::vector<Ort::AllocatedStringPtr> output_names_ptr;

        std::vector<std::vector<int64_t>> inputShapes;
        std::vector<std::vector<int64_t>> outputShapes;
        float confThreshold = 0.3f;
        float iouThreshold = 0.4f;

        bool hasMask = false;
        float maskThreshold = 0.5f;
};