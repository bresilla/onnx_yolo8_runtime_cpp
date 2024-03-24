#include "inference.hpp"

namespace utils {
	std::vector<cv::Point> get_contour(const cv::Mat& mask, bool join){
		cv::Mat mask2 = mask.clone();
		cv::Mat mask8;
		mask2.convertTo(mask8, CV_8UC1, 255);
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(mask8, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		if (join){
			std::vector<cv::Point> contour;
			for (int i = 0; i < contours.size(); i++){
				contour.insert(contour.end(), contours[i].begin(), contours[i].end());
			}
			return contour;
		}
		else{
			return contours[0];
		}
	}

	void visualizeDetection(cv::Mat &im, std::vector<Detection> &results, const std::vector<std::string> &classNames) {
		cv::Mat image = im.clone();
		for (const Detection &result : results) {
			int x = result.bbox.x;
			int y = result.bbox.y;
			int conf = (int)std::round(result.accu * 100);
			int classId = result.id;
			std::string label = classNames[classId] + " 0." + std::to_string(conf);
			int baseline = 0;
			cv::Size size = cv::getTextSize(label, cv::FONT_ITALIC, 0.4, 1, &baseline);
			image(result.bbox).setTo(colors[classId + classNames.size()], result.mask);
			cv::rectangle(image, result.bbox, colors[classId], 2);
			cv::rectangle(image, cv::Point(x, y), cv::Point(x + size.width, y + 12), colors[classId], -1);
			cv::putText(image, label, cv::Point(x, y - 3 + 12), cv::FONT_ITALIC, 0.4, cv::Scalar(0, 0, 0), 1);
		}
		cv::addWeighted(im, 0.4, image, 0.6, 0, im);
	}

	void letterbox(const cv::Mat &image, cv::Mat &outImage, const cv::Size &newShape = cv::Size(640, 640), 
						const cv::Scalar &color = cv::Scalar(114, 114, 114), bool auto_ = true, bool scaleFill = false, bool scaleUp = true, int stride = 32) {
		cv::Size shape = image.size();
		float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
		if (!scaleUp) { r = std::min(r, 1.0f); }

		float ratio[2]{r, r};
		int newUnpad[2]{(int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r)};

		auto dw = (float)(newShape.width - newUnpad[0]);
		auto dh = (float)(newShape.height - newUnpad[1]);

		if (auto_) {
			dw = (float)((int)dw % stride);
			dh = (float)((int)dh % stride);
		} else if (scaleFill) {
			dw = 0.0f;
			dh = 0.0f;
			newUnpad[0] = newShape.width;
			newUnpad[1] = newShape.height;
			ratio[0] = (float)newShape.width / (float)shape.width;
			ratio[1] = (float)newShape.height / (float)shape.height;
		}

		dw /= 2.0f;
		dh /= 2.0f;

		if (shape.width != newUnpad[0] && shape.height != newUnpad[1]) {
			cv::resize(image, outImage, cv::Size(newUnpad[0], newUnpad[1]));
		}

		int top = int(std::round(dh - 0.1f));
		int bottom = int(std::round(dh + 0.1f));
		int left = int(std::round(dw - 0.1f));
		int right = int(std::round(dw + 0.1f));
		cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
	}
}

void utils::scaleCoords(cv::Rect &coords, cv::Mat &mask, const float maskThreshold, const cv::Size &imageShape, const cv::Size &imageOriginalShape) {
    float gain = std::min((float)imageShape.height / (float)imageOriginalShape.height, (float)imageShape.width / (float)imageOriginalShape.width);

    int pad[2] = {(int)(((float)imageShape.width - (float)imageOriginalShape.width * gain) / 2.0f), (int)(((float)imageShape.height - (float)imageOriginalShape.height * gain) / 2.0f)};

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.x = std::max(0, coords.x);
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));
    coords.y = std::max(0, coords.y);

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.width = std::min(coords.width, imageOriginalShape.width - coords.x);
    coords.height = (int)std::round(((float)coords.height / gain));
    coords.height = std::min(coords.height, imageOriginalShape.height - coords.y);
    mask = mask(cv::Rect(pad[0], pad[1], imageShape.width - 2 * pad[0], imageShape.height - 2 * pad[1]));

    cv::resize(mask, mask, imageOriginalShape, cv::INTER_LINEAR);
    mask = mask(coords) > maskThreshold;
}



ONNXInf::ONNXInf(const std::string &modelPath, const bool &isGPU, float confThreshold, float iouThreshold, float maskThreshold) {
    this->confThreshold = confThreshold;
    this->iouThreshold = iouThreshold;
    this->maskThreshold = maskThreshold;
    env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "YOLOV8");
    sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    if (isGPU && (cudaAvailable == availableProviders.end())){
        std::cout << "GPU is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        std::cout << "Inference device: CPU" << std::endl;
    }
    else if (isGPU && (cudaAvailable != availableProviders.end())){
        std::cout << "Inference device: GPU" << std::endl;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    }
    else {
        std::cout << "Inference device: CPU" << std::endl;
    }

    session = Ort::Session(env, modelPath.c_str(), sessionOptions);

    const size_t num_input_nodes = session.GetInputCount();   //==1
    const size_t num_output_nodes = session.GetOutputCount(); //==1,2
    if (num_output_nodes > 1) {
        this->hasMask = true;
        std::cout << "Instance Segmentation" << std::endl;
    }
    else
        std::cout << "Object Detection" << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        this->inputNames.push_back(input_name.get());
        input_names_ptr.push_back(std::move(input_name));

        Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(i);
        std::vector<int64_t> inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->inputShapes.push_back(inputTensorShape);
        this->isDynamicInputShape = false;
        // checking if width and height are dynamic
        if (inputTensorShape[2] == -1 && inputTensorShape[3] == -1) {
            std::cout << "Dynamic input shape" << std::endl;
            this->isDynamicInputShape = true;
        }
    }
    for (int i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        this->outputNames.push_back(output_name.get());
        output_names_ptr.push_back(std::move(output_name));

        Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(i);
        std::vector<int64_t> outputTensorShape = outputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        this->outputShapes.push_back(outputTensorShape);
        if (i == 0) {
            if (!this->hasMask)
                classNums = outputTensorShape[1] - 4;
            else
                classNums = outputTensorShape[1] - 4 - 32;
        }
    }
}

void ONNXInf::getBestClassInfo(std::vector<float>::iterator it, float &bestConf, int &bestClassId, const int _classNums){
    // first 4 element are box
    bestClassId = 4;
    bestConf = 0;
    for (int i = 4; i < _classNums + 4; i++) {
        if (it[i] > bestConf) {
            bestConf = it[i];
            bestClassId = i - 4;
        }
    }
}

cv::Mat ONNXInf::getMask(const cv::Mat &maskProposals, const cv::Mat &maskProtos){
    cv::Mat protos = maskProtos.reshape(0, {(int)this->outputShapes[1][1], (int)this->outputShapes[1][2] * (int)this->outputShapes[1][3]});
    cv::Mat matmul_res = (maskProposals * protos).t();
    cv::Mat masks = matmul_res.reshape(1, {(int)this->outputShapes[1][2], (int)this->outputShapes[1][3]});
    cv::Mat dest;

    // sigmoid
    cv::exp(-masks, dest);
    dest = 1.0 / (1.0 + dest);
    cv::resize(dest, dest, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]), cv::INTER_LINEAR);
    return dest;
}

void ONNXInf::preprocessing(cv::Mat &image, float *&blob, std::vector<int64_t> &inputTensorShape){
    cv::Mat resizedImage, floatImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    utils::letterbox(resizedImage, resizedImage, cv::Size((int)this->inputShapes[0][2], (int)this->inputShapes[0][3]), cv::Scalar(114, 114, 114), this->isDynamicInputShape, false, true, 32);

    inputTensorShape[2] = resizedImage.rows;
    inputTensorShape[3] = resizedImage.cols;

    resizedImage.convertTo(floatImage, CV_32FC3, 1 / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{floatImage.cols, floatImage.rows};
    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i) {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

std::vector<Detection> ONNXInf::postprocessing(const cv::Size &resizedImageShape, const cv::Size &originalImageShape, std::vector<Ort::Value> &outputTensors){
    // for box
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> classIds;

    float *boxOutput = outputTensors[0].GetTensorMutableData<float>();
    //[1,4+n,8400]=>[1,8400,4+n] or [1,4+n+32,8400]=>[1,8400,4+n+32]
    cv::Mat output0 = cv::Mat(cv::Size((int)this->outputShapes[0][2], (int)this->outputShapes[0][1]), CV_32F, boxOutput).t();
    float *output0ptr = (float *)output0.data;
    int rows = (int)this->outputShapes[0][2];
    int cols = (int)this->outputShapes[0][1];
    // std::cout << rows << cols << std::endl;
    // if hasMask
    std::vector<std::vector<float>> picked_proposals;
    cv::Mat mask_protos;

    for (int i = 0; i < rows; i++)
    {
        std::vector<float> it(output0ptr + i * cols, output0ptr + (i + 1) * cols);
        float confidence;
        int classId;
        this->getBestClassInfo(it.begin(), confidence, classId, classNums);

        if (confidence > this->confThreshold)
        {
            if (this->hasMask)
            {
                std::vector<float> temp(it.begin() + 4 + classNums, it.end());
                picked_proposals.push_back(temp);
            }
            int centerX = (int)(it[0]);
            int centerY = (int)(it[1]);
            int width = (int)(it[2]);
            int height = (int)(it[3]);
            int left = centerX - width / 2;
            int top = centerY - height / 2;
            boxes.emplace_back(left, top, width, height);
            confs.emplace_back(confidence);
            classIds.emplace_back(classId);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, this->confThreshold, this->iouThreshold, indices);

    if (this->hasMask) {
        float *maskOutput = outputTensors[1].GetTensorMutableData<float>();
        std::vector<int> mask_protos_shape = {1, (int)this->outputShapes[1][1], (int)this->outputShapes[1][2], (int)this->outputShapes[1][3]};
        mask_protos = cv::Mat(mask_protos_shape, CV_32F, maskOutput);
    }

    std::vector<Detection> results;
    for (int idx : indices) {
        Detection res;
        res.bbox = cv::Rect(boxes[idx]);
        if (this->hasMask)
            res.mask = this->getMask(cv::Mat(picked_proposals[idx]).t(), mask_protos);
        else
            res.mask = cv::Mat::zeros((int)this->inputShapes[0][2], (int)this->inputShapes[0][3], CV_8U);

		res.contour = utils::get_contour(res.mask, false);

        utils::scaleCoords(res.bbox, res.mask, this->maskThreshold, resizedImageShape, originalImageShape);
        res.accu = confs[idx];
        res.id = classIds[idx];
        results.emplace_back(res);
    }
    return results;
}

std::vector<Detection> ONNXInf::predict(cv::Mat &image) {
    float *blob = nullptr;
    std::vector<int64_t> inputTensorShape{1, 3, -1, -1};
    this->preprocessing(image, blob, inputTensorShape);
    size_t inputTensorSize = this->vectorProduct(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));
    std::vector<Ort::Value> outputTensors = this->session.Run(Ort::RunOptions{nullptr}, this->inputNames.data(), inputTensors.data(), 1, this->outputNames.data(), this->outputNames.size());
    cv::Size resizedShape = cv::Size((int)inputTensorShape[3], (int)inputTensorShape[2]);
    std::vector<Detection> result = this->postprocessing(resizedShape, image.size(), outputTensors);
    delete[] blob;
    return result;
}


size_t ONNXInf::vectorProduct(const std::vector<int64_t> &vector){
    if (vector.empty()) { return 0;} 
    size_t product = 1;
    for (const auto &element : vector) {
        product *= element;
    }
    return product;
}