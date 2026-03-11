import torch
import torch.nn.functional as F
import numpy as np

class MetricLogger:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        
    def update(self, preds, targets):
        """
        Update confusion matrix with new predictions and targets.
        preds: (B, H, W) tensor or numpy array
        targets: (B, H, W) tensor or numpy array
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
            
        preds = preds.flatten()
        targets = targets.flatten()
        
        # Filter out ignore_index (if any, e.g. 255)
        # Assuming 0 to num_classes-1 are valid classes
        mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]
        
        if len(targets) > 0:
            self.confusion_matrix += np.bincount(
                self.num_classes * targets.astype(int) + preds.astype(int),
                minlength=self.num_classes ** 2
            ).reshape(self.num_classes, self.num_classes)
            
    def compute(self):
        """
        Compute metrics from the accumulated confusion matrix.
        Returns a dictionary with OA, mIoU, mDice, and per-class IoU.
        """
        # OA
        oa = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + 1e-6)
        
        # mIoU
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-6)
        miou = np.nanmean(iou)
        
        # Dice (per class) = 2*TP / (2*TP + FP + FN) = 2*Intersection / (Sum_Pred + Sum_Target)
        # Sum_Pred = sum(axis=0), Sum_Target = sum(axis=1)
        dice = 2 * intersection / (self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) + 1e-6)
        mdice = np.nanmean(dice)
        
        return {"OA": oa, "mIoU": miou, "mDice": mdice, "IoU": iou}
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

def calculate_semantic_segmentation(outputs, target_size):
    """
    Convert UperHead outputs (logits) to semantic segmentation map.
    outputs: Logits tensor of shape [B, C, H, W]
    target_size: Not needed anymore as upsampling is done in the model's forward pass,
                 but kept for API consistency if needed elsewhere.
    
    Returns:
        pred_label: (B, H, W) tensor with class indices
    """
    # The 'outputs' from SimpleUperHead are already logits of shape [B, C, H, W]
    # We just need to find the class with the highest score for each pixel.
    pred_label = torch.argmax(outputs, dim=1) # (B, H, W)
    
    return pred_label
