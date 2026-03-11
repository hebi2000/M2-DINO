import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(1) + targets.sum(1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    # Flatten spatial dimensions to ensure mean(1) averages over all pixels
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 12544):
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice
        self.num_points = num_points

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_mask = outputs["pred_masks"].flatten(0, 1)  # [batch_size * num_queries, H_pred, W_pred]

        # Also concat the target labels and masks
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_mask = torch.cat([v["masks"] for v in targets])

        # Compute the classification cost.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between masks
        # We downsample target masks to prediction size for cost computation
        H_pred, W_pred = out_mask.shape[-2:]
        tgt_mask_scaled = F.interpolate(tgt_mask.unsqueeze(1).float(), size=(H_pred, W_pred), mode="nearest").squeeze(1)
        
        out_mask = out_mask.sigmoid()
        
        # Flatten spatial dimensions
        out_mask_flat = out_mask.flatten(1)
        tgt_mask_flat = tgt_mask_scaled.flatten(1)

        # Clamp for numerical stability
        out_mask_flat = out_mask_flat.clamp(1e-6, 1 - 1e-6)

        # Calculate BCE cost using matrix multiplication
        pos_cost = -(torch.log(out_mask_flat) @ tgt_mask_flat.t())
        neg_cost = -(torch.log(1 - out_mask_flat) @ (1 - tgt_mask_flat).t())
        cost_mask = (pos_cost + neg_cost) / (H_pred * W_pred)
        
        # Compute the dice cost (Pairwise)
        intersection = torch.mm(out_mask_flat, tgt_mask_flat.t())
        p_sum = out_mask_flat.sum(1).unsqueeze(1)
        t_sum = tgt_mask_flat.sum(1).unsqueeze(0)
        cost_dice = 1 - (2 * intersection + 1) / (p_sum + t_sum + 1)

        # Final cost matrix
        C = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["labels"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = F.interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                  mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0]

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (tgt, _) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

# Helper for nested tensors (simplified)
class NestedTensor(object):
    def __init__(self, tensors, mask=None):
        self.tensors = tensors
        self.mask = mask
    def decompose(self):
        return self.tensors, self.mask

def nested_tensor_from_tensor_list(tensor_list: list):
    if tensor_list[0].ndim == 3:
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = (len(tensor_list),) + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)

def is_dist_avail_and_initialized():
    return False
def get_world_size():
    return 1

def prepare_targets(masks, num_classes, ignore_value=255):
    """
    Convert semantic mask (B, H, W) to list of dicts for Mask2Former.
    Each dict: {'labels': (N,), 'masks': (N, H, W)}
    """
    targets = []
    for mask in masks:
        # mask: (H, W) with class IDs
        labels = torch.unique(mask)
        
        labels_list = []
        masks_list = []
        for label in labels:
            # Filter out ignore_value
            if label.item() == ignore_value:
                continue
                
            # Create binary mask for this class
            binary_mask = (mask == label).float()
            labels_list.append(label)
            masks_list.append(binary_mask)
            
        if len(labels_list) > 0:
            target = {
                "labels": torch.stack(labels_list).long(),
                "masks": torch.stack(masks_list),
            }
        else:
            target = {
                "labels": torch.tensor([], dtype=torch.long, device=mask.device),
                "masks": torch.tensor([], dtype=torch.float, device=mask.device).reshape(0, mask.shape[0], mask.shape[1]),
            }
        targets.append(target)
    return targets
