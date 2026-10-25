"""Build the detector: TorchVision FCOS-ResNet50-FPN, plus a tiny variant for tests.

Why FCOS rather than Faster R-CNN:

- It is single-stage and anchor-free: a convolutional backbone and FPN, then
  dense conv heads and one NMS. Faster R-CNN adds a region proposal stage,
  per-image proposal sampling and RoIAlign, which are more per-image Python
  work, scale less cleanly with batch size, and are more likely to hit an
  under-tested ROCm kernel.
- It needs about half the compute per image of Faster R-CNN v2 in
  TorchVision's published figures, which matters for real-time throughput on
  one consumer GPU and for fine-tuning in 16 GB.
- Faster R-CNN v2 is more accurate on COCO. FCOS is accurate enough for cars
  and pedestrians at intersection scale, and fine-tuning on the crossing
  viewpoint matters more than the architecture; switching later only means
  adding a builder here.

TorchVision detection models keep class 0 for background, so project class
``i`` (0-based in the config) is label ``i + 1``.
"""

from __future__ import annotations

import math
from collections import OrderedDict

from crosswalk.config import ModelConfig

# COCO category names in TorchVision's 91-slot label order ("N/A" marks unused
# ids). Kept here so baseline evaluation does not need the weights metadata.
COCO_CATEGORIES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]  # fmt: skip


def label_names(config: ModelConfig) -> list[str]:
    """Names indexed by label id, including background at 0."""
    if config.classes:
        return ["__background__", *config.classes]
    return list(COCO_CATEGORIES)


def _replace_classification_head(model, num_classes: int) -> None:
    """Swap FCOS's final class conv for one sized to the project classes.

    The backbone, FPN, box regression head and classification tower keep their
    COCO-pretrained weights; only the last 3x3 conv is new. It is initialised
    the way TorchVision initialises it (small weights, bias set so every class
    starts at 1% probability) so focal loss is stable from the first step.
    """
    from torch import nn

    head = model.head.classification_head
    old = head.cls_logits
    num_anchors = old.out_channels // head.num_classes
    new = nn.Conv2d(old.in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    nn.init.normal_(new.weight, std=0.01)
    prior = 0.01
    nn.init.constant_(new.bias, -math.log((1 - prior) / prior))
    head.cls_logits = new
    head.num_classes = num_classes


class TinyBackbone:
    """Factory for a two-level, 32-channel backbone (built lazily to keep torch optional)."""

    @staticmethod
    def build():
        from torch import nn

        class _Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                # FCOS's head uses GroupNorm(32, channels), so 32 channels is
                # the smallest width that works.
                self.out_channels = 32
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                )  # fmt: skip
                self.down = nn.Sequential(nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())

            def forward(self, x):
                c1 = self.stem(x)
                c2 = self.down(c1)
                return OrderedDict([("0", c1), ("1", c2)])

        return _Tiny()


def build_detector(config: ModelConfig, *, for_training: bool = False):
    """Return an FCOS model configured for inference or fine-tuning."""
    from torchvision.models.detection import FCOS, fcos_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    project_classes = len(config.classes) + 1 if config.classes else None
    common = {
        "min_size": config.min_size,
        "max_size": config.max_size,
        "score_thresh": config.score_threshold,
    }

    if config.name == "fcos_tiny":
        anchors = AnchorGenerator(sizes=((16,), (32,)), aspect_ratios=((1.0,), (1.0,)))
        return FCOS(
            TinyBackbone.build(),
            num_classes=project_classes or len(COCO_CATEGORIES),
            anchor_generator=anchors,
            topk_candidates=200,
            detections_per_img=50,
            **common,
        )

    if config.weights == "COCO_V1":
        from torchvision.models.detection import FCOS_ResNet50_FPN_Weights

        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=config.trainable_backbone_layers if for_training else None,
            **common,
        )
        if project_classes is not None:
            _replace_classification_head(model, project_classes)
        return model

    # Random initialisation (the GPU check): no download needed.
    return fcos_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=project_classes or len(COCO_CATEGORIES),
        **common,
    )


def describe_weights(config: ModelConfig) -> str:
    """Short model version string for metrics and detection output."""
    if config.checkpoint:
        return config.checkpoint
    return f"{config.name}:{config.weights}"
