import onnxruntime as ort
import numpy as np
import cv2


class OnnxPredictorPython:
    def __init__(self, model_path: str, use_gpu: bool = True):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # 获取输入尺寸
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 缩放到模型输入尺寸
        resized_image = cv2.resize(image_rgb, (self.input_width, self.input_height))
        # 归一化到 [0, 1] 并转换为 float32
        image_float = resized_image.astype(np.float32) / 255.0
        # 标准化到 [-1, 1]
        normalized_image = (image_float - 0.5) / 0.5
        # HWC -> NCHW (Height, Width, Channel -> Batch, Channel, Height, Width)
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        # 增加 batch 维度
        input_tensor = np.expand_dims(transposed_image, axis=0)
        return input_tensor

    def predict_clas(self, image: np.ndarray) -> int:
        input_tensor = self._preprocess(image)

        # 运行推理
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # 后处理
        scores = outputs[0][0]  # 获取第一个输出的第一个批次
        predicted_class_id = np.argmax(scores)

        return int(predicted_class_id)


# --- 使用示例 ---
if __name__ == "__main__":
    predictor = OnnxPredictorPython("path/to/your/model.onnx", use_gpu=True)

    # 读入一张图片
    # image = cv2.imread("path/to/your/image.jpg")
    image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)  # 示例图片

    class_id = predictor.predict_clas(image)
    print(f"Predicted Class ID: {class_id}")
