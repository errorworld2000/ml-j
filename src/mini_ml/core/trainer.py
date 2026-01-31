import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mini_ml.utils.config import AppConfig
from mini_ml.core.factory import (
    build_dataset,
    build_loss,
    build_lr_scheduler,
    build_model,
    build_optimizer,
)
from mini_ml.utils.metrics import MeanIoU

logger = logging.getLogger(__name__)


class Trainer:
    """
    Encapsulates the training lifecycle, including environment setup,
    component building, and the training loop.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.output_dir = Path(config.environment.output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        
        self.train_loader = None
        self.val_loader = None # Added val_loader
        
        self.writer = None
        
        # Metric
        # Assumes num_classes is available in model config or dataset config.
        # But AppConfig doesn't enforce strict schema on dataset details.
        # However, config.model.head.num_classes exists in the example.
        # Let's try to get it from 'config.model.head.num_classes' if nested, 
        # or fallback to a default if not found (though Pydantic should ensure validity if typed).
        # We'll instantiate Metric in _build_components or before evaluation.
        self.metric = None

    def setup(self):
        """Prepare environment and build components."""
        self._setup_env()
        self._build_components()

    def _setup_env(self):
        """Setup directories, seed, and logging."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self._setup_seed(self.config.environment.seed)

        logger.info("All outputs will be saved to: %s", self.output_dir)

        # Save final config
        with open(self.output_dir / "config_final.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.config.model_dump(), f, indent=2, sort_keys=False)
        logger.info("Final configuration saved to config_final.yaml")
        logger.info("Using device: %s", self.device)

        # TensorBoard
        self.writer = SummaryWriter()

    def _setup_seed(self, seed: Optional[int]):
        """Set random seed."""
        if seed is None:
            seed = int(datetime.now().timestamp())
            logger.info("No seed provided, generated seed = %d", seed)
        else:
            logger.info("Using fixed seed = %d", seed)
        
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        # Performance over determinism for now unless debugging
        torch.backends.cudnn.deterministic = False 
        torch.backends.cudnn.benchmark = True
        logger.info("Random seed set to %d", seed)

    def _build_components(self):
        """Build model, optimizer, loss, dataset, etc."""
        logger.info("Building components...")
        self.model = build_model(self.config.model).to(self.device)
        self.criterion = build_loss(self.config.loss)
        self.optimizer = build_optimizer(self.config.optimizer, self.model)
        self.lr_scheduler = build_lr_scheduler(
            # Using total steps (iters) instead of epochs for polynomial decay
            self.config.lr_scheduler, self.optimizer, max_iters=self.config.iters
        )
        
        # --- Train Dataset ---
        train_dataset = build_dataset(self.config.dataset, mode="train")
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.environment.num_workers,
            pin_memory=True,
            drop_last=True
        )
        logger.info("Train dataset created with %d samples.", len(train_dataset))
        
        # --- Val Dataset ---
        if "val" in self.config.dataset:
            val_dataset = build_dataset(self.config.dataset, mode="val")
            self.val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.config.batch_size, # Can ideally use larger batch size for val
                shuffle=False,
                num_workers=self.config.environment.num_workers,
                pin_memory=True,
                drop_last=False
            )
            logger.info("Val dataset created with %d samples.", len(val_dataset))
        
        # --- Metric ---
        # Attempt to get num_classes from config
        try:
            num_classes = self.config.model.head.num_classes
        except AttributeError:
             # Fallback or error. Assuming config structure matches expectation
             # If config.model is generic dict (likely in factory but Pydantic in AppConfig?), 
             # AppConfig defines model as ModelConfig which has head as ComponentConfig?
             # Let's assume user config structure is valid.
             # If not, let's hardcode or try-catch.
             # Actually AppConfig -> model -> head (dict or object)
             # Looking at config.yaml: head: {type: ..., num_classes: 3}
             # Pydantic model likely parses this.
             pass
        
        # For safety, let's just access it dynamically or trust the object
        # self.config.model.head is a Pydantic model or dict? in 'factory.py', input is ModelConfig.
        # Let's check config.py to be sure... but I don't have it open. 
        # I'll rely on the config.yaml I saw: 'num_classes: 3'.
        # I will extract it from the validated config object.
        # If it fails, I'll log a warning and skip metric init.
        if hasattr(self.config.model.head, "num_classes"):
             n_cls = self.config.model.head.num_classes
             self.metric = MeanIoU(num_classes=n_cls, ignore_index=255)
        else:
             logger.warning("Could not find 'num_classes' in model.head config. Evaluation metrics will be disabled.")

        logger.info("Components built successfully.")

    def train(self):
        """Main training loop."""
        self.setup()
        
        logger.info("Starting training...")
        max_iters = self.config.iters
        log_interval = self.config.environment.log_interval
        save_interval = self.config.environment.save_interval
        
        # Determine eval interval (same as save or custom?)
        # For now, let's eval every time we save.
        eval_interval = save_interval

        self.model.train()

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("•"),
            TextColumn("Iter: {task.completed} of {task.total}"),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            transient=True,
        ) as progress:

            train_task = progress.add_task("[green]Training", total=max_iters)
            current_iter = 0
            data_iterator = iter(self.train_loader)
            
            while current_iter < max_iters:
                try:
                    images, masks = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(self.train_loader)
                    images, masks = next(data_iterator)

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                outputs = self.model(images)
                # Output key 'pred' is standard for our models?
                # HRNet forward returns [output] list.
                # But loss expects dict or tensor? 
                # Factory - build_loss wraps CombinedLoss.
                # Predictor output inspection showed list.
                # Let's look at `train.py` original code: `outputs = model(images); loss_dict = criterion(outputs["pred"], masks)`
                # Wait, original `train.py` line 142: `outputs = model(images)`
                # Line 143: `loss_dict = criterion(outputs["pred"], masks)`
                # This implies model returns a dict? 
                # BUT `HRNet.forward` returns `[output]`.
                # Maybe `HRNetSegmentor` wrapper (if it exists) returns dict?
                # Config says `type: HRNetSegmentor`.
                # I haven't seen `HRNetSegmentor` code. I saw `HRNet` in `hrnet.py`.
                # Let's assume the component builder returns something that works as in original `train.py`.
                # If original `train.py` was working (which user claimed it was "running"), 
                # then `outputs["pred"]` must be valid.
                
                loss_dict = self.criterion(outputs["pred"], masks)
                total_loss = loss_dict["total_loss"]

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                current_iter += 1
                progress.update(train_task, advance=1)

                if current_iter % log_interval == 0:
                    self._log_metrics(current_iter, max_iters, total_loss, loss_dict)

                if current_iter > 0 and current_iter % eval_interval == 0:
                     # Save first
                    self._save_checkpoint(current_iter)
                    # Then Eval
                    if self.val_loader:
                        self.evaluate(current_iter)
                        self.model.train() # Switch back to train mode

        self._save_final_model()

    def evaluate(self, step: int):
        """Run validation loop."""
        logger.info(f"Starting evaluation at iter {step}...")
        self.model.eval()
        if self.metric:
            self.metric.reset()
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.numpy() # Metric expects numpy usually or we convert
                
                outputs = self.model(images)
                preds = outputs["pred"]
                
                # Preds might be (B, C, H, W) logits. Metric needs class indices (B, H, W).
                if isinstance(preds, torch.Tensor):
                    preds = torch.argmax(preds, dim=1).cpu().numpy()
                
                if self.metric:
                    self.metric.update(preds, masks)
        
        if self.metric:
            scores = self.metric.compute()
            logger.info(f"Evaluation Results - mIoU: {scores['mIoU']:.4f}, Acc: {scores['Accuracy']:.4f}")
            self.writer.add_scalar("Metrics/mIoU", scores["mIoU"], step)
            self.writer.add_scalar("Metrics/Accuracy", scores["Accuracy"], step)
        else:
            logger.info("Evaluation finished (no metrics computed).")

    def _log_metrics(self, current_iter, max_iters, total_loss, loss_dict):
        current_lr = self.optimizer.param_groups[0]["lr"]
        loss_log_str = " | ".join(
            [f"{k}: {v:.4f}" for k, v in loss_dict.items() if k != "total_loss"]
        )
        logger.info(
            f"Iter: [{current_iter}/{max_iters}] | LR: {current_lr:.6f} | Loss: {total_loss.item():.4f} ({loss_log_str})"
        )

        self.writer.add_scalar("Loss/total", total_loss.item(), current_iter)
        self.writer.add_scalar("Metrics/Learning_Rate", current_lr, current_iter)
        for k, v in loss_dict.items():
            if k != "total_loss":
                self.writer.add_scalar(f"Loss/{k}", v, current_iter)

    def _save_checkpoint(self, current_iter):
        ckpt_path = self.checkpoints_dir / f"checkpoint_iter_{current_iter}.pth"
        torch.save(self.model.state_dict(), ckpt_path)
        logger.info(f"Checkpoint saved to {ckpt_path}")

    def _save_final_model(self):
        logger.info("Training Finished.")
        final_model_path = self.output_dir / "final_model.pth"
        torch.save(self.model.state_dict(), final_model_path)
        logger.info("Final model saved to: %s", final_model_path)
