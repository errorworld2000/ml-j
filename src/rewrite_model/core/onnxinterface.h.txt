#ifndef ONNXINTERFACE_H
#define ONNXINTERFACE_H

#include <opencv.hpp>
#include <QString>

struct det_rec {
	int class_id;
	float confidence;
	cv::Rect bbox;
};

class OnnxInterface
{
public:
	virtual ~OnnxInterface() {}
	virtual bool LoadNet(QString modelPath, bool useGPU) = 0;
	virtual void UnloadNet() = 0;
	virtual void predict_image_seg(cv::Mat src, cv::Mat& dst) = 0;
	virtual std::vector<det_rec> predict_image_det(cv::Mat src) = 0;
	virtual int predict_image_clas(cv::Mat src) = 0;
};

class OnnxFactory
{
public:
	virtual ~OnnxFactory() {}

	virtual OnnxInterface* createInstance() = 0;
};

QT_BEGIN_NAMESPACE

#define OnnxInterface_iid "xx.plugin.Onnx"
#define OnnxFactory_iid "xx.plugin.OnnxFactory"

Q_DECLARE_INTERFACE(OnnxInterface, OnnxInterface_iid)
Q_DECLARE_INTERFACE(OnnxFactory, OnnxFactory_iid)

QT_END_NAMESPACE

#endif // ONNXINTERFACE_H
