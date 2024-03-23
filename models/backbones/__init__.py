#from .head import IQAHead, VARHead, VQAHead, MaxVQAHead,simpleVQAHead
from .swin_backbone import SwinTransformer2D as IQABackbone
from .swin_backbone import SwinTransformer3D as VQABackbone
from .swin_backbone import swin_3d_small, swin_3d_tiny


__all__ = [
    "VQABackbone",
    "IQABackbone",
    "VQAHead",
    "MaxVQAHead",
    "IQAHead",
    "VARHead",
    "simpleVQAHead",
    "BaseEvaluator",
    "BaseImageEvaluator",
    "DOVER",
    "resnet50"
]
