import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

__all__ = [
    "BaseMaskRCNNHead",
    "AlfredHead",
    "build_reachable_head",
    "ROI_ALFRED_HEAD_REGISTRY",
]


ROI_ALFRED_HEAD_REGISTRY = Registry("ROI_ALFRED_HEAD")
ROI_ALFRED_HEAD_REGISTRY.__doc__ = """
Registry for alfred heads, which predicts instance masks given
per-region features.
The registered object will be called with `obj(cfg, input_shape)`.
"""


@torch.jit.unused
def alfred_reachable_loss(pred_reachable_logits: torch.Tensor, instances: List[Instances]):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    gt_classes = []
    gt_reachables = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
        gt_classes.append(gt_classes_per_image)

        gt_reachables_per_image = instances_per_image.gt_reachables.to(dtype=torch.bool)
        gt_reachables.append(gt_reachables_per_image)

    if len(gt_reachables) == 0:
        return pred_reachable_logits.sum() * 0

    indices = torch.arange(pred_reachable_logits.size(0))
    gt_classes = cat(gt_classes, dim=0)
    gt_reachables = cat(gt_reachables, dim=0)
    pred_reachable_logits = pred_reachable_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.5 threshold)
    reachable_incorrect = (pred_reachable_logits > 0.0) != gt_reachables
    reachable_accuracy = 1 - (
        reachable_incorrect.sum().item() / max(reachable_incorrect.numel(), 1.0)
    )
    num_positive = gt_reachables.sum().item()
    false_positive = (reachable_incorrect & ~gt_reachables).sum().item() / max(
        gt_reachables.numel() - num_positive, 1.0
    )
    false_negative = (reachable_incorrect & gt_reachables).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("alfred_reachable/accuracy", reachable_accuracy)
    storage.put_scalar("alfred_reachable/false_positive", false_positive)
    storage.put_scalar("alfred_reachable/false_negative", false_negative)

    reachable_loss = F.binary_cross_entropy_with_logits(
        pred_reachable_logits, gt_reachables * 1.0, reduction="mean"
    )
    return reachable_loss


def alfred_reachable_inference(
    pred_reachable_logits: torch.Tensor, pred_instances: List[Instances]
):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """

    # Select masks corresponding to the predicted classes
    num_reachables = pred_reachable_logits.shape[0]
    class_pred = cat([i.pred_classes for i in pred_instances])
    indices = torch.arange(num_reachables, device=class_pred.device)
    reachable_probs_pred = pred_reachable_logits[indices, class_pred].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    reachable_probs_pred = reachable_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(reachable_probs_pred, pred_instances):
        instances.pred_reachables = prob  # (1, Hmask, Wmask)


@torch.jit.unused
def mask_rcnn_loss(pred_mask_logits: torch.Tensor, instances: List[Instances], vis_period: int = 0):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.
    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def mask_rcnn_inference(pred_mask_logits: torch.Tensor, pred_instances: List[Instances]):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.
    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".
    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic described in :paper:`Mask R-CNN`
    """

    @configurable
    def __init__(self, *, loss_weight: float = 1.0, vis_period: int = 0):
        """
        NOTE: this interface is experimental.
        Args:
            loss_weight (float): multiplier of the loss
            vis_period (int): visualization period
        """
        super().__init__()
        self.vis_period = vis_period
        self.loss_weight = loss_weight

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {"vis_period": cfg.VIS_PERIOD}

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period) * self.loss_weight}
        else:
            mask_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


# To get torchscript support, we make the head a subclass of `nn.Sequential`.
# Therefore, to add new layers in this head class, please make sure they are
# added in the order they will be used in forward().
@ROI_ALFRED_HEAD_REGISTRY.register()
class AlfredHead(nn.Sequential):
    """
    A alfred head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        *,
        num_classes: int,
        conv_dims: List[int],
        fc_dims: List[int],
        conv_norm="",
        loss_weight: float = 1.0,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of foreground classes (i.e. background is not
                included). 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__()
        self.loss_weight = loss_weight
        assert len(conv_dims) + len(fc_dims) > 0, "conv_dims have to be non-empty!"

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=nn.ReLU(),
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.add_module("flatten", nn.Flatten())
            fc = nn.Linear(int(np.prod(self._output_size)), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.add_module("fc_relu{}".format(k + 1), nn.ReLU())
            self.fcs.append(fc)
            self._output_size = fc_dim

        # input_size = (
        #     self._output_size.channels
        #     * (self._output_size.width or 1)
        #     * (self._output_size.height or 1)
        # )
        self.predictor = nn.Linear(self._output_size, num_classes)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        # use normal distribution initialization for reachable prediction layer
        nn.init.normal_(self.predictor.weight, std=0.01)
        nn.init.constant_(self.predictor.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_ALFRED_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_ALFRED_HEAD.NUM_CONV
        fc_dim = cfg.MODEL.ROI_ALFRED_HEAD.FC_DIM
        num_fc = cfg.MODEL.ROI_ALFRED_HEAD.NUM_FC
        ret = {
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "conv_dims": [conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_ALFRED_HEAD.NORM,
            "input_shape": input_shape,
        }
        return ret

    def layers(self, x):
        for layer in self:
            x = layer(x)
        return x

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.
        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x = self.layers(x)
        if self.training:
            return {"loss_reachable": alfred_reachable_loss(x, instances) * self.loss_weight}
        else:
            alfred_reachable_inference(x, instances)
            return instances


def build_reachable_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_ALFRED_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_ALFRED_HEAD.NAME
    return ROI_ALFRED_HEAD_REGISTRY.get(name)(cfg, input_shape)
