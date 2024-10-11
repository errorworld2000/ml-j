#ifndef ONNXPLUGIN_H
#define ONNXPLUGIN_H

#include <QObject>
#include <QtPlugin>
#include "../onnxinterface.h"

#include "onnxruntime_cxx_api.h"
#include <opencv.hpp>


class OnnxPlugin : public QObject, public OnnxInterface
{
	Q_OBJECT
		//Q_PLUGIN_METADATA(IID OnnxInterface_iid)
		Q_INTERFACES(OnnxInterface)
public:
	explicit OnnxPlugin(QObject* parent = nullptr);
	~OnnxPlugin() {}
	void predict_image_seg(cv::Mat src, cv::Mat& dst);
	std::vector<det_rec> predict_image_det(cv::Mat src);
	bool LoadNet(QString modelPath, bool useGPU);
	void UnloadNet();
	int predict_image_clas(cv::Mat src);
signals:

public slots:

private:
	std::vector<int64> m_input_dims;
	std::vector<int64> m_output_dims;

	cv::Mat preprocess(cv::Mat image);
	bool m_bOpen;
	Ort::Env m_env;
	Ort::RunOptions m_run_options{ NULL };
	Ort::Session m_session{ NULL };
};
class OnnxPluginFactory : public QObject, public OnnxFactory
{
	Q_OBJECT
		Q_PLUGIN_METADATA(IID OnnxFactory_iid)
		Q_INTERFACES(OnnxFactory)

public:
	OnnxInterface* createInstance() {
		return new OnnxPlugin();
	}
};
#endif // ONNXPLUGIN_H
