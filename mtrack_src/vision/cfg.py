from detectron2.config import CfgNode as CN


def add_alfred_config(cfg: CN):
    _C = cfg
    _C.MODEL.ROI_ALFRED_HEAD = CN()
    _C.MODEL.ROI_ALFRED_HEAD.NAME = "AlfredHead"
    _C.MODEL.ROI_ALFRED_HEAD.POOLER_RESOLUTION = 14
    _C.MODEL.ROI_ALFRED_HEAD.POOLER_SAMPLING_RATIO = 0
    _C.MODEL.ROI_ALFRED_HEAD.NUM_CONV = 0  # The number of convs in the mask head
    _C.MODEL.ROI_ALFRED_HEAD.CONV_DIM = 256
    # Normalization method for the convolution layers.
    # Options: "" (no norm), "GN", "SyncBN".
    _C.MODEL.ROI_ALFRED_HEAD.NORM = ""
    # Type of pooling operation applied to the incoming feature map for each RoI
    _C.MODEL.ROI_ALFRED_HEAD.POOLER_TYPE = "ROIAlignV2"
    _C.MODEL.ROI_ALFRED_HEAD.NUM_FC = 0
    # Hidden layer dimension for FC layers in the RoI box head
    _C.MODEL.ROI_ALFRED_HEAD.FC_DIM = 1024
