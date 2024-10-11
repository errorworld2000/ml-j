#include "onnxplugin.h"

OnnxPlugin::OnnxPlugin(QObject* parent) : QObject(parent), m_bOpen(false)
{
}

int OnnxPlugin::predict_image_clas(cv::Mat src)
{
	if (!m_bOpen)
		return -1;
	cv::Mat preprocessed_image = preprocess(src);

	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size(m_input_dims[3], m_input_dims[2]), cv::Scalar(0, 0, 0), false, true);
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// size_t siz = blob.total();
	//    std::vector<int64_t> input_shape;
	//    for (int i : m_input_shape) {
	//        int64_t tmp = static_cast<int64_t>(i);
	//        input_shape.push_back(tmp);
	//    }

	std::vector<Ort::Value> input_tensors;
	input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, blob.ptr<float>(), blob.total(), m_input_dims.data(), m_input_dims.size()));

	// Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_tensor_size, input_shape.data(), input_shape.size());

	Ort::AllocatorWithDefaultOptions allocator;

	std::vector<const char*> out_node_names;
	std::vector<const char*> input_node_names;

	Ort::AllocatedStringPtr input_name = m_session.GetInputNameAllocated(0, allocator);
	input_node_names.push_back(input_name.get());

	Ort::AllocatedStringPtr output_name = m_session.GetOutputNameAllocated(0, allocator);
	out_node_names.push_back(output_name.get());

	std::vector<Ort::Value> output_tensors = m_session.Run(
		Ort::RunOptions{ NULL },
		input_node_names.data(),
		input_tensors.data(),
		input_node_names.size(),
		out_node_names.data(),
		out_node_names.size());
	float* floatarr = output_tensors[0].GetTensorMutableData<float>();
	// int64* floatarr = output_tensors_[0].GetTensorMutableData<int64>();

	//    int out_num = std::accumulate(m_input_shape.begin(), m_input_shape.end(), 1,
	//        std::multiplies<int>());
	//    Ort::TypeInfo type_info = m_session.GetOutputTypeInfo(0);
	//    Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
	//    std::vector<int64_t> output_dims = tensor_info.GetShape();
	int out_num = std::accumulate(m_output_dims.begin(), m_output_dims.end(), 1, std::multiplies<int>());

	std::vector<float> out_data_u8(out_num);
	float max = 0;
	int index = 0;
	// #pragma omp parallel for
	for (int i = 0; i < out_num; i++)
	{
		out_data_u8[i] = static_cast<float>(floatarr[i]);
		if (out_data_u8[i] > max)
		{
			max = out_data_u8[i];
			index = i;
		}
	}
	input_tensors.clear();
	output_tensors.clear();

	return index;
}

void OnnxPlugin::predict_image_seg(cv::Mat src, cv::Mat& dst)
{
	if (!m_bOpen)
		return;
	cv::Mat preprocessed_image = preprocess(src);

	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size(m_input_dims[3], m_input_dims[2]), cv::Scalar(0, 0, 0), false, true);
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

	// size_t siz = blob.total();
	//    std::vector<int64_t> input_shape;
	//    for (int i : m_input_shape) {
	//        int64_t tmp = static_cast<int64_t>(i);
	//        input_shape.push_back(tmp);
	//    }

	std::vector<Ort::Value> input_tensors;
	input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, blob.ptr<float>(), blob.total(), m_input_dims.data(), m_input_dims.size()));

	// Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_tensor_size, input_shape.data(), input_shape.size());

	Ort::AllocatorWithDefaultOptions allocator;

	std::vector<const char*> out_node_names;
	std::vector<const char*> input_node_names;

	Ort::AllocatedStringPtr input_name = m_session.GetInputNameAllocated(0, allocator);
	input_node_names.push_back(input_name.get());

	Ort::AllocatedStringPtr output_name = m_session.GetOutputNameAllocated(0, allocator);
	out_node_names.push_back(output_name.get());

	std::vector<Ort::Value> output_tensors = m_session.Run(
		Ort::RunOptions{ NULL },
		input_node_names.data(),
		input_tensors.data(),
		input_node_names.size(),
		out_node_names.data(),
		out_node_names.size());
	int* floatarr = output_tensors[0].GetTensorMutableData<int>();
	// int64* floatarr = output_tensors_[0].GetTensorMutableData<int64>();

	//    int out_num = std::accumulate(m_input_shape.begin(), m_input_shape.end(), 1,
	//        std::multiplies<int>());
	//    Ort::TypeInfo type_info = m_session.GetOutputTypeInfo(0);
	//    Ort::ConstTensorTypeAndShapeInfo tensor_info = type_info.GetTensorTypeAndShapeInfo();
	//    std::vector<int64_t> output_dims = tensor_info.GetShape();
	int out_num = std::accumulate(m_output_dims.begin(), m_output_dims.end(), 1, std::multiplies<int>());

	std::vector<uchar> out_data_u8(out_num);
#pragma omp parallel for
	for (int i = 0; i < out_num; i++)
	{
		out_data_u8[i] = static_cast<uchar>(floatarr[i]);
	}

	// cv::Mat mask = cv::Mat::zeros(static_cast<int>(input_node_dims_[2]), static_cast<int>(input_node_dims_[3]), CV_8UC1);
	cv::Mat mask(m_input_dims[2], m_input_dims[3], CV_8UC1, out_data_u8.data());

	cv::resize(mask, mask, src.size());
	input_tensors.clear();
	output_tensors.clear();
	dst = mask.clone();
	return;
}

std::vector<det_rec> OnnxPlugin::predict_image_det(cv::Mat src)
{
	if (!m_bOpen)
		return {};
	cv::Mat preprocessed_image = preprocess(src);
	cv::Mat blob = cv::dnn::blobFromImage(preprocessed_image, 1, cv::Size(m_input_dims[3], m_input_dims[2]), cv::Scalar(0, 0, 0), false, true);
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<Ort::Value> input_tensors;
	std::vector<float> im_shape = { static_cast<float>(preprocessed_image.rows), static_cast<float>(preprocessed_image.cols) };
	std::vector<float> scale_factor = { static_cast<float>(preprocessed_image.cols) / src.cols, static_cast<float>(preprocessed_image.rows) / src.rows };
	input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, im_shape.data(), im_shape.size(), std::array<int64_t, 2>{1, 2}.data(), 2));
	input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, blob.ptr<float>(), blob.total(), m_input_dims.data(), m_input_dims.size()));
	input_tensors.emplace_back(Ort::Value::CreateTensor(memory_info, scale_factor.data(), scale_factor.size(), std::array<int64_t, 2>{1, 2}.data(), 2));
	Ort::AllocatorWithDefaultOptions allocator;
	std::vector<std::string> input_node_names;
	size_t numInputNodes = m_session.GetInputCount();
	for (size_t i = 0; i < numInputNodes; i++) {
		auto input_name_ptr = m_session.GetInputNameAllocated(i, allocator);
		if (input_name_ptr) {
			input_node_names.emplace_back(input_name_ptr.get());
		}
		else {
			std::cerr << "Failed to get input name for node " << i << std::endl;
			return {};
		}
	}

	std::vector<std::string> output_node_names;
	size_t numOutputNodes = m_session.GetOutputCount();
	for (size_t i = 0; i < numOutputNodes; i++) {
		auto output_name_ptr = m_session.GetOutputNameAllocated(i, allocator);
		if (output_name_ptr) {
			output_node_names.emplace_back(output_name_ptr.get());
		}
		else {
			std::cerr << "Failed to get output name for node " << i << std::endl;
			return {};
		}
	}

	std::vector<const char*> input_names_cstr;
	std::vector<const char*> output_names_cstr;

	for (const auto& name : input_node_names) {
		input_names_cstr.push_back(name.c_str());
	}

	for (const auto& name : output_node_names) {
		output_names_cstr.push_back(name.c_str());
	}

	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = m_session.Run(
			Ort::RunOptions{ nullptr },
			input_names_cstr.data(),
			input_tensors.data(),
			input_names_cstr.size(),
			output_names_cstr.data(),
			output_names_cstr.size());
	}
	catch (const Ort::Exception& e) {
		std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl;
		return {};
	}
	catch (const std::exception& e) {
		std::cerr << "Standard Exception: " << e.what() << std::endl;
		return {};
	}

	Ort::Value& output0 = ort_outputs[0];
	auto output0_shape = output0.GetTensorTypeAndShapeInfo().GetShape();
	int num_detections = output0_shape[0];
	int feature_size = output0_shape[1];
	const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
	cv::Mat dout(num_detections, feature_size, CV_32F, (float*)pdata);
	cv::Mat det_output = dout;
	//cv::Mat det_output = dout.t(); 

	//Ort::Value& output1 = ort_outputs[1];
	//auto output1_shape = output1.GetTensorTypeAndShapeInfo().GetShape();
	//int num_classes = output1_shape[0];
	//int32_t* output1_data = ort_outputs[1].GetTensorMutableData<int32_t>();
	std::vector<det_rec> result;
	for (int i = 0; i < det_output.rows; i++) {
		auto scores = det_output.row(i).colRange(1, 2);
		cv::Point classIdPoint;
		double maxScore;
		minMaxLoc(scores, nullptr, &maxScore, nullptr, &classIdPoint);
		if (maxScore > 0.25) {
			int x = static_cast<int>(det_output.at<float>(i, 2));
			int y = static_cast<int>(det_output.at<float>(i, 3));
			int width = static_cast<int>(det_output.at<float>(i, 4) - det_output.at<float>(i, 2));
			int height = static_cast<int>(det_output.at<float>(i, 5) - det_output.at<float>(i, 3));
			det_rec det{ classIdPoint.x, static_cast<float>(maxScore), cv::Rect(x, y, width, height) };
			result.push_back(det);
		}
	}

	return result;
}

bool OnnxPlugin::LoadNet(QString modelPath, bool useGPU)
{
	Ort::SessionOptions sessionOptions;
	if (useGPU)
	{
		OrtCUDAProviderOptions options;
		options.device_id = 0;
		options.arena_extend_strategy = 0;
		options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::OrtCudnnConvAlgoSearchExhaustive;
		options.do_copy_in_default_stream = 1;
		options.gpu_mem_limit = (size_t)4 * 1024 * 1024 * 1024;

		sessionOptions.AppendExecutionProvider_CUDA(options);
	}

	unsigned int num_threads = std::thread::hardware_concurrency();
	sessionOptions.SetIntraOpNumThreads(num_threads);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	// Ort::Env env;// (ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
	try
	{
		m_session = Ort::Session(m_env, modelPath.toStdWString().c_str(), sessionOptions);
	}
	catch (...)
	{
		m_bOpen = false;
		return false;
	}

	/*Ort::AllocatorWithDefaultOptions allocator;

	Ort::AllocatedStringPtr input_name = session_.GetInputNameAllocated(0, allocator);
	input_node_names_.push_back(input_name.get());

	Ort::AllocatedStringPtr output_name = session_.GetOutputNameAllocated(0, allocator);
	out_node_names_.push_back(output_name.get());*/

	for (size_t i = 0; i < m_session.GetInputCount(); ++i) {
		Ort::TypeInfo input_type_info = m_session.GetInputTypeInfo(i);
		Ort::ConstTensorTypeAndShapeInfo input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		std::vector<int64_t> input_dims = input_tensor_info.GetShape();
		// 检查维度大小是否为 4
		if (input_dims.size() == 4) {
			m_input_dims = input_dims; 
			break; 
		}
	}

	Ort::TypeInfo output_type_info = m_session.GetOutputTypeInfo(0);
	Ort::ConstTensorTypeAndShapeInfo output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
	m_output_dims = output_tensor_info.GetShape();
	if (m_input_dims[0] < 0)
		m_input_dims[0] = 1;
	if (m_output_dims[0] < 0)
		m_output_dims[0] = 1;
	m_bOpen = true;
	return true;
}

void OnnxPlugin::UnloadNet()
{
	m_session.release();
	m_bOpen = false;
	return;
}

cv::Mat OnnxPlugin::preprocess(cv::Mat image)
{
	cv::Mat dst, dst_float; // , normalized_image;
	if (image.channels() == 1)
		cv::cvtColor(image, dst, cv::COLOR_GRAY2RGB);
	else
		cv::cvtColor(image, dst, cv::COLOR_BGR2RGB);

	cv::resize(dst, dst, cv::Size(m_input_dims[3], m_input_dims[2]), 0, 0);
	dst.convertTo(dst_float, CV_32F, 1.0 / 255.0);

	dst_float = (dst_float - 0.5) / 0.5;

	return dst_float;
}
