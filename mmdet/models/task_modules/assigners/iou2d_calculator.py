# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps, get_box_tensor


def cast_tensor_type(x, scale=1., dtype=None):
    if dtype == 'fp16':
        # scale is for preventing overflows
        x = (x / scale).half()
    return x


@TASK_UTILS.register_module()
class BboxOverlaps2D:
    """2D Overlaps (e.g. IoUs, GIoUs) Calculator."""

    def __init__(self, scale=1., dtype=None):
        self.scale = scale
        self.dtype = dtype

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        """Calculate IoU between 2D bboxes.

        Args:
            bboxes1 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, or shape (m, 5) in <x1, y1, x2,
                y2, score> format.
            bboxes2 (Tensor or :obj:`BaseBoxes`): bboxes have shape (m, 4)
                in <x1, y1, x2, y2> format, shape (m, 5) in <x1, y1, x2, y2,
                score> format, or be empty. If ``is_aligned `` is ``True``,
                then m and n must be equal.
            mode (str): "iou" (intersection over union), "iof" (intersection
                over foreground), or "giou" (generalized intersection over
                union).
            is_aligned (bool, optional): If True, then m and n must be equal.
                Default False.

        Returns:
            Tensor: shape (m, n) if ``is_aligned `` is False else shape (m,)
        """
        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)
        assert bboxes1.size(-1) in [0, 4, 5]
        assert bboxes2.size(-1) in [0, 4, 5]
        if bboxes2.size(-1) == 5:
            bboxes2 = bboxes2[..., :4]
        if bboxes1.size(-1) == 5:
            bboxes1 = bboxes1[..., :4]

        if self.dtype == 'fp16':
            # change tensor type to save cpu and cuda memory and keep speed
            bboxes1 = cast_tensor_type(bboxes1, self.scale, self.dtype)
            bboxes2 = cast_tensor_type(bboxes2, self.scale, self.dtype)
            overlaps = bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
            if not overlaps.is_cuda and overlaps.dtype == torch.float16:
                # resume cpu float32
                overlaps = overlaps.float()
            return overlaps

        return bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + f'(' \
            f'scale={self.scale}, dtype={self.dtype})'
        return repr_str


@TASK_UTILS.register_module()
class BboxOverlaps2D_GLIP(BboxOverlaps2D):

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        TO_REMOVE = 1
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + TO_REMOVE) * (
            bboxes1[:, 3] - bboxes1[:, 1] + TO_REMOVE)
        area2 = (bboxes2[:, 2] - bboxes2[:, 0] + TO_REMOVE) * (
            bboxes2[:, 3] - bboxes2[:, 1] + TO_REMOVE)

        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [N,M,2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou


@TASK_UTILS.register_module()
class RotatedBboxOverlaps2D:
    """2D IoU/IoF calculator for rotated boxes.

    Boxes use ``(cx, cy, w, h, angle)`` format. Tensors with six values per
    row are accepted for compatibility with detection heads that append a
    score; the last value is ignored.

    Args:
        angle_unit (str): ``'radian'`` or ``'degree'``.
        clockwise (bool): Positive-angle direction convention.
    """

    def __init__(self,
                 angle_unit: str = 'radian',
                 clockwise: bool = True) -> None:
        if angle_unit not in ('radian', 'degree'):
            raise ValueError("angle_unit must be either 'radian' or 'degree', "
                             f'but got {angle_unit!r}.')
        self.angle_unit = angle_unit
        self.clockwise = clockwise

    def __call__(self,
                 bboxes1,
                 bboxes2,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6):
        bboxes1 = get_box_tensor(bboxes1)
        bboxes2 = get_box_tensor(bboxes2)
        if bboxes1.size(-1) not in (5, 6) or bboxes2.size(-1) not in (5, 6):
            raise ValueError(
                'RotatedBboxOverlaps2D expects boxes with 5 or 6 values '
                f'per row, but got {tuple(bboxes1.shape)} and '
                f'{tuple(bboxes2.shape)}.')
        bboxes1 = bboxes1[..., :5]
        bboxes2 = bboxes2[..., :5]
        from mmdet.models.layers.rotated_box_iou import box_iou_rotated
        return box_iou_rotated(
            bboxes1,
            bboxes2,
            mode=mode,
            aligned=is_aligned,
            clockwise=self.clockwise,
            angle_unit=self.angle_unit,
            eps=eps)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'angle_unit={self.angle_unit!r}, '
                f'clockwise={self.clockwise})')


RBboxOverlaps2D = RotatedBboxOverlaps2D
