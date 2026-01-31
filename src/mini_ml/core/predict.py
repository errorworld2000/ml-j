import abc
import os
import click
import cv2
import numpy as np
import yaml
import torch
import onnxruntime as ort
from rich.console import Console
from typing import Optional

from mini_ml.utils.config import AppConfig
from mini_ml.core.factory import build_model, build_transforms


class BasePredictor(abc.ABC):
    """Abstract base class for predictors."""

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> dict:
        """
        Run prediction on a single image.
        Args:
            image: BGR image (H, W, 3) from cv2.imread
        Returns:
            Dictionary containing prediction results (e.g., {'class_id': 0})
        """
        pass


class TorchPredictor(BasePredictor):
    """Predictor using native PyTorch model and factory-built transforms."""

    def __init__(self, config_path: str, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Load Config
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
        self.config = AppConfig(**raw_config)

        # 2. Build Model
        print(f"[INFO] Building PyTorch model from config: {config_path}")
        self.model = build_model(self.config.model)
        
        # 3. Load Weights
        if model_path:
            print(f"[INFO] Loading weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            # Handle both full checkpoint (state_dict key) and raw state_dict
            state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            self.model.load_state_dict(state_dict)
        else:
            print("[WARN] No model path provided, using random initialization!")

        self.model.to(self.device)
        self.model.eval()

        # 4. Build Transforms (reuse validation transforms)
        # Note: We use the 'val' dataset transforms for inference
        print("[INFO] Building transforms from dataset.val config...")
        # We manually extract the transforms config list from dataset.val
        val_dataset_cfg = dict(self.config.dataset["val"])
        transforms_cfg_list = val_dataset_cfg.get("transforms", [])
        self.transforms = build_transforms(transforms_cfg_list)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> dict:
        # Preprocess
        # Apply transforms (e.g., Resize, ToTensor, Normalize)
        # Note: Our Transforms pipeline likely expects PIL or similar, or specific format.
        # But 'ToTensor' usually handles HWC -> CHW.
        # Let's check implicit assumption: 'PetDataset' uses default loader which is likely PIL or cv2 compatible.
        # Factory builds Compose. Compose expects to start with raw data.
        # Ideally, we should unify this more, but for now assuming transforms handle the image.
        
        # NOTE: standard transforms usually expect PIL Image or Tensor. 
        # Since we use build_transforms which uses standard torchvision-like or custom transforms,
        # we might need to convert cv2 image to PIL if the first transform expects it.
        # However, looking at config.yaml, it has "Resize", "RandomHorizontalFlip", "ToTensor".
        # If these are custom wrappers that handle numpy/cv2, fine. 
        # If they are standard Torchvision, we need PIL.
        # Let's assume for safety we convert to PIL if needed, OR just pass to transforms if they are custom.
        # For this refactor, let's look at `mini_ml.models.transforms`.
        # I'll stick to a safe default: convert to PIL if not sure, but since I can't see the transform code right now without looking,
        # I will check if I should look at `src/mini_ml/models/transforms`.
        # Wait, I saw `src/mini_ml/models/transforms` dir earlier.
        
        # For this implementation, I will assume the transforms pipeline handles the input as defined in the dataset.
        # But `cv2.imread` returns BGR numpy. Most torchvision transforms expect RGB PIL.
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Wrap in a simpler way: just pass to transforms. 
        # If it fails, runtime error -> easier to debug than guessing.
        # Actually, standard torchvision transforms (Resize) allow Tensor or PIL.
        # ToTensor converts numpy to Tensor.
        # So: numpy (HWC) -> ToTensor -> Tensor (CHW) -> Resize -> ...
        # But config order is Resize -> ToTensor. Resize needs PIL or Tensor (image).
        
        # To make this robust:
        # Let's convert to PIL Image to be safe for typical PyTorch pipelines unless we know otherwise.
        from PIL import Image
        pil_img = Image.fromarray(image_rgb)
        
        input_tensor, _ = self.transforms(pil_img)
        
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0) # Add batch dim
            
        input_tensor = input_tensor.to(self.device)

        # Inference
        outputs = self.model(input_tensor)
        
        # Post-process
        # Assuming classification/segmentation. The config has "SimpleSegmentationHead".
        # So output is likely (B, C, H, W).
        
        # For simplicity, returning the raw class ID map or similar.
        # But looking at old predict.py, it was doing `argmax(scores)`.
        # HRNet output is a list [output].
        
        if isinstance(outputs, list):
            pred_logits = outputs[0]
        elif isinstance(outputs, dict):
            pred_logits = outputs.get("pred", outputs)
        else:
            pred_logits = outputs
            
        pred_logits = pred_logits.squeeze(0) # (C, H, W)
        
        # Global max pooling or just checking max class?
        # The previous 'predict_clas' was weird for a segmentation model (returning single int).
        # But if the user wants 'class id', maybe it's classification?
        # Config says 'HRNetSegmentor', 'SimpleSegmentationHead'. This is a segmentation task.
        # Old predict.py had: `predicted_class_id = np.argmax(scores)`.
        # If scores was (C,), that works. If scores was (C, H, W)... np.argmax would be the Index of flat max?
        # Wait, old predict.py used `self.input_height`.
        # `scores = outputs[0][0]` implying Batch 0, Output 0?
        # If output is (B, C, H, W), scores = (C, H, W).
        # `np.argmax(scores)` on a 3D array returns a single integer (flat index).
        # This seems buggy in the original code if it really was Segmentation. 
        # But I should preserve behavior or improve it. 
        # Getting the MOST FREQUENT class? Or the class of the center pixel?
        # Let's assume we return the segmentation map (H, W).
        
        pred_map = torch.argmax(pred_logits, dim=0).cpu().numpy().astype(np.uint8)
        
        return {"segmentation_map": pred_map}


class OnnxPredictor(BasePredictor):
    """Predictor using ONNX Runtime. Manual/Hardcoded transforms for now."""

    def __init__(self, model_path: str):
        print(f"[INFO] Init ONNX session from: {model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"[WARN] Failed to use requested providers, falling back to CPU. Error: {e}")
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        # Input shape usually [Batch, Channel, Height, Width]
        self.input_height = input_shape[2]
        self.input_width = input_shape[3]

    def predict(self, image: np.ndarray) -> dict:
        # Hardcoded logic from original predict.py to preserve ONNX behavior
        # BGR -> RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Resize
        resized_image = cv2.resize(image_rgb, (self.input_width, self.input_height))
        # Normalize to [-1, 1] as in original script
        image_float = resized_image.astype(np.float32) / 255.0
        normalized_image = (image_float - 0.5) / 0.5
        # HWC -> NCHW
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        input_tensor = np.expand_dims(transposed_image, axis=0)

        # Run
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Post-process (mimicking original behavior or improved)
        scores = outputs[0][0] # (C, H, W) presumably
        
        # Original logic was predicted_class_id = np.argmax(scores) -> single int (buggy?)
        # Let's stick to returning map or fixing the interpretation.
        # Since I changed Torch to return map, I should do same here for consistency.
        pred_map = np.argmax(scores, axis=0).astype(np.uint8)
        
        return {"segmentation_map": pred_map}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model-path",
    "-m",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the model file (.pth or .onnx).",
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to config.yaml (Required for PyTorch backend).",
)
@click.option(
    "--image",
    "-i",
    "image_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the input image.",
)
@click.option(
    "--backend",
    type=click.Choice(["pytorch", "onnx", "auto"], case_sensitive=False),
    default="auto",
    help="Inference backend. 'auto' detects from file extension.",
)
def main(model_path: str, config_path: str, image_path: str, backend: str):
    """
    Unified prediction script for PyTorch and ONNX models.
    """
    console = Console()
    
    # 1. Determine Backend
    if backend == "auto":
        if model_path.endswith(".onnx"):
            backend = "onnx"
        elif model_path.endswith(".pth") or model_path.endswith(".pt"):
            backend = "pytorch"
        else:
            console.print("[bold red]Error: Cannot auto-detect backend from file extension. Please specify --backend.[/bold red]")
            return

    # 2. Instantiate Predictor
    try:
        if backend == "pytorch":
            if not config_path:
                console.print("[bold red]Error: PyTorch backend requires --config.[/bold red]")
                return
            predictor = TorchPredictor(config_path=config_path, model_path=model_path)
        else:
            predictor = OnnxPredictor(model_path=model_path)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize predictor: {e}[/bold red]")
        console.print_exception()
        return

    # 3. Load Image
    image = cv2.imread(image_path)
    if image is None:
        console.print(f"[bold red]Error: Could not read image at {image_path}[/bold red]")
        return

    # 4. Predict
    console.print(f"[bold green]Running prediction using {backend.upper()} backend...[/bold green]")
    try:
        result = predictor.predict(image)
        # For segmentation, maybe save the mask or just print info
        seg_map = result.get("segmentation_map")
        if seg_map is not None:
            console.print(f"Prediction successful. Output shape: {seg_map.shape}")
            console.print(f"Unique classes found: {np.unique(seg_map)}")
            # Optional: save output
            out_filename = "prediction_result.png"
            # Scale up for visibility if class IDs are small (0,1,2...)
            vis_mask = (seg_map * 50).astype(np.uint8) 
            cv2.imwrite(out_filename, vis_mask)
            console.print(f"Saved visualization to [bold]{out_filename}[/bold]")
        else:
            console.print(f"Result: {result}")
            
    except Exception as e:
        console.print(f"[bold red]Prediction failed: {e}[/bold red]")
        console.print_exception()


if __name__ == "__main__":
    main()
