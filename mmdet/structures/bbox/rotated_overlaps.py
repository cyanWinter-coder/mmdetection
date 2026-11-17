# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .base_boxes import BaseBoxes


def _get_tensor(boxes):
    if isinstance(boxes, BaseBoxes):
        boxes = boxes.tensor
    if boxes.size(-1) == 6:
        boxes = boxes[..., :5]
    return boxes


def box_iou_rotated(boxes1,
                    boxes2,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True,
                    angle_unit: str = 'radian',
                    eps: float = 1e-6) -> Tensor:
    """Structure-level wrapper for rotated-box IoU/IoF."""
    from mmdet.models.layers.rotated_box_iou import box_iou_rotated as overlap
    return overlap(
        _get_tensor(boxes1),
        _get_tensor(boxes2),
        mode=mode,
        aligned=aligned,
        clockwise=clockwise,
        angle_unit=angle_unit,
        eps=eps)


def rbbox_overlaps(boxes1,
                   boxes2,
                   mode: str = 'iou',
                   is_aligned: bool = False,
                   **kwargs) -> Tensor:
    """MMRotate-compatible rotated-box IoU/IoF wrapper."""
    return box_iou_rotated(
        boxes1, boxes2, mode=mode, aligned=is_aligned, **kwargs)


__all__ = ['box_iou_rotated', 'rbbox_overlaps']
