import numpy as np


class MeanIoU:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        Update confusion matrix.
        Args:
            preds: (N, H, W) or (N, ...) numpy array of predicted class indices.
            targets: (N, H, W) or (N, ...) numpy array of target class indices.
        """
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        
        # Calculate confusion matrix for this batch
        # bincount trick for fast confusion matrix calculation
        # x = preds +  targets * num_classes
        # This maps each pair (pred, target) to a unique integer
        n = self.num_classes
        if len(preds) > 0:
            x = preds + n * targets
            bincount = np.bincount(x, minlength=n**2)
            cm = bincount.reshape((n, n))
            self.confusion_matrix += cm

    def compute(self) -> dict:
        """
        Compute mIoU and Accuracy.
        Returns:
            dict containing 'mIoU', 'Accuracy', and per-class IoU.
        """
        # IoU = TP / (TP + FP + FN)
        # TP = diag(CM)
        # FP+FN+TP = sum(CM, axis=1) + sum(CM, axis=0) - diag(CM)
        
        cm = self.confusion_matrix
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        intersection = tp
        union = tp + fp + fn
        
        # Avoid division by zero
        # If union is 0, IoU is technically undefined (or 1 if TP is also 0, or 0)
        # Usually we treat it as nan and ignore in mean, or 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = intersection / union
            
        # Accuracy = Total TP / Total Pixels
        total_pixels = cm.sum()
        accuracy = tp.sum() / total_pixels if total_pixels > 0 else 0.0
        
        # Mean IoU (ignoring NaN classes which were not present in target at all)
        # A class present in target but never predicted will have FN > 0 -> Union > 0 -> IoU = 0.
        # A class never present in target and never predicted will have Union = 0 -> IoU = NaN.
        valid_mask = ~np.isnan(ious)
        m_iou = np.mean(ious[valid_mask]) if valid_mask.any() else 0.0
        
        return {
            "mIoU": m_iou,
            "Accuracy": accuracy,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
