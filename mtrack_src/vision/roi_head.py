import inspect
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import ImageList, Instances
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
    select_foreground_proposals,
)

from .alfred_head import build_reachable_head


@ROI_HEADS_REGISTRY.register()
class AlfredStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(
        self,
        reachable_in_features: Optional[List[str]] = None,
        reachable_pooler: Optional[ROIPooler] = None,
        reachable_head: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.reachable_in_features = reachable_in_features
        self.reachable_pooler = reachable_pooler
        self.reachable_head = reachable_head

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if inspect.ismethod(cls._init_reachable_head):
            ret.update(cls._init_reachable_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_reachable_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_ALFRED_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_ALFRED_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_ALFRED_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"reachable_in_features": in_features}
        ret["reachable_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["reachable_head"] = build_reachable_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training:
            proposals, losses = super().forward(images, features, proposals, targets)
            losses.update(self._forward_reachable(features, proposals))
            return proposals, losses
        else:
            pred_instancs, _ = super().forward(images, features, proposals, targets)
            self._forward_reachable(features, pred_instancs)
            return pred_instancs, {}

    def _forward_reachable(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.
        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.reachable_pooler is not None:
            features = [features[f] for f in self.reachable_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.reachable_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.reachable_in_features}
        return self.reachable_head(features, instances)
