# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Sequence, Union

import torch
from torch import Tensor

from mmdet.registry import TASK_UTILS
from mmdet.structures.bbox import BaseBoxes, get_box_tensor
from .base_bbox_coder import BaseBBoxCoder


def _norm_angle(angle: Tensor, angle_version: str) -> Tensor:
    if angle_version == 'oc':
        return angle
    if angle_version == 'le135':
        return torch.remainder(angle + math.pi / 4, math.pi) - math.pi / 4
    if angle_version == 'le90':
        return torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    if angle_version == 'r360':
        return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi
    raise ValueError(
        "angle_version must be one of 'oc', 'le90', 'le135' or 'r360', "
        f'but got {angle_version!r}.')


@TASK_UTILS.register_module()
class DeltaXYWHTRBBoxCoder(BaseBBoxCoder):
    """Encode and decode rotated boxes in ``(cx, cy, w, h, angle)`` format.

    Args:
        target_means (Sequence[float]): Target means for the five deltas.
        target_stds (Sequence[float]): Target standard deviations.
        angle_version (str): Angle representation used for targets.
        norm_factor (float, optional): Optional factor applied to angle
            deltas. A value of ``pi`` is common in oriented detectors.
        edge_swap (bool): Whether to choose the smaller equivalent angle by
            swapping width and height.
        proj_xy (bool): Whether to project center deltas into the anchor
            frame.
        use_box_type (bool): Return :class:`RotatedBoxes` from ``decode``.
    """

    encode_size = 5

    def __init__(self,
                 target_means: Sequence[float] = (0., 0., 0., 0., 0.),
                 target_stds: Sequence[float] = (1., 1., 1., 1., 1.),
                 angle_version: str = 'le90',
                 norm_factor: Optional[float] = None,
                 edge_swap: bool = False,
                 proj_xy: bool = False,
                 use_box_type: bool = False,
                 **kwargs) -> None:
        super().__init__(use_box_type=use_box_type, **kwargs)
        if len(target_means) != 5 or len(target_stds) != 5:
            raise ValueError(
                'target_means and target_stds must both contain 5 values, '
                f'but got {target_means} and {target_stds}.')
        if angle_version not in ('oc', 'le90', 'le135', 'r360'):
            raise ValueError(
                "angle_version must be one of 'oc', 'le90', 'le135' or "
                f"'r360', but got {angle_version!r}.")
        self.means = target_means
        self.stds = target_stds
        self.angle_version = angle_version
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy

    def encode(self, bboxes: Union[Tensor, BaseBoxes],
               gt_bboxes: Union[Tensor, BaseBoxes]) -> Tensor:
        bboxes = get_box_tensor(bboxes).float()
        gt_bboxes = get_box_tensor(gt_bboxes).float()
        if bboxes.size() != gt_bboxes.size() or bboxes.size(-1) != 5:
            raise ValueError(
                'encode requires equally shaped rotated boxes with last '
                f'dimension 5, but got {tuple(bboxes.shape)} and '
                f'{tuple(gt_bboxes.shape)}.')

        px, py, pw, ph, pt = bboxes.unbind(dim=-1)
        gx, gy, gw, gh, gt = gt_bboxes.unbind(dim=-1)
        if (pw <= 0).any() or (ph <= 0).any():
            raise ValueError(
                'encode received a non-positive proposal width or height. '
                f'Proposal shape: {tuple(bboxes.shape)}.')

        if self.proj_xy:
            delta_x = (torch.cos(pt) * (gx - px) + torch.sin(pt) *
                       (gy - py)) / pw
            delta_y = (-torch.sin(pt) * (gx - px) + torch.cos(pt) *
                       (gy - py)) / ph
        else:
            delta_x = (gx - px) / pw
            delta_y = (gy - py) / ph

        if self.edge_swap:
            delta_angle_1 = _norm_angle(gt - pt, self.angle_version)
            delta_angle_2 = _norm_angle(gt - pt + math.pi / 2,
                                        self.angle_version)
            choose_first = delta_angle_1.abs() < delta_angle_2.abs()
            gt_width = torch.where(choose_first, gw, gh)
            gt_height = torch.where(choose_first, gh, gw)
            delta_angle = torch.where(choose_first, delta_angle_1,
                                      delta_angle_2)
        else:
            gt_width, gt_height = gw, gh
            delta_angle = _norm_angle(gt - pt, self.angle_version)

        delta_w = torch.log(gt_width.clamp_min(1e-7) / pw)
        delta_h = torch.log(gt_height.clamp_min(1e-7) / ph)
        if self.norm_factor:
            delta_angle = delta_angle / (self.norm_factor * math.pi)

        deltas = torch.stack([delta_x, delta_y, delta_w, delta_h, delta_angle],
                             dim=-1)
        means = deltas.new_tensor(self.means)
        stds = deltas.new_tensor(self.stds)
        return (deltas - means) / stds

    def decode(
        self,
        bboxes: Union[Tensor, BaseBoxes],
        pred_bboxes: Tensor,
        max_shape: Optional[Union[Sequence[int], Tensor,
                                  Sequence[Sequence[int]]]] = None,
        wh_ratio_clip: Optional[float] = 16 / 1000
    ) -> Union[Tensor, BaseBoxes]:
        proposals = get_box_tensor(bboxes)
        if proposals.size(-1) != 5 or pred_bboxes.size(-1) != 5:
            raise ValueError(
                'decode requires rotated boxes and deltas with last '
                f'dimension 5, but got {tuple(proposals.shape)} and '
                f'{tuple(pred_bboxes.shape)}.')
        if proposals.size(0) != pred_bboxes.size(0):
            raise ValueError(
                'decode requires matching first dimensions, but got '
                f'{proposals.size(0)} and {pred_bboxes.size(0)}.')
        if wh_ratio_clip is None or wh_ratio_clip <= 0:
            wh_ratio_clip = 16 / 1000

        means = pred_bboxes.new_tensor(self.means)
        stds = pred_bboxes.new_tensor(self.stds)
        deltas = pred_bboxes * stds + means
        delta_x, delta_y, delta_w, delta_h, delta_angle = deltas.unbind(dim=-1)

        ratio_limit = abs(math.log(wh_ratio_clip))
        width = proposals[..., 2] * torch.exp(
            delta_w.clamp(-ratio_limit, ratio_limit))
        height = proposals[..., 3] * torch.exp(
            delta_h.clamp(-ratio_limit, ratio_limit))
        center_x = proposals[..., 0] + delta_x * proposals[..., 2]
        center_y = proposals[..., 1] + delta_y * proposals[..., 3]
        angle = proposals[..., 4] + delta_angle
        if self.norm_factor:
            angle = angle * (self.norm_factor * math.pi)
        angle = _norm_angle(angle, self.angle_version)

        if self.edge_swap:
            choose_first = torch.abs(
                _norm_angle(angle - proposals[..., 4],
                            self.angle_version)) < torch.abs(
                                _norm_angle(
                                    angle - proposals[..., 4] + math.pi / 2,
                                    self.angle_version))
            width, height = (torch.where(choose_first, width, height),
                             torch.where(choose_first, height, width))
            angle = torch.where(
                choose_first, angle,
                _norm_angle(angle + math.pi / 2, self.angle_version))

        if max_shape is not None and not isinstance(max_shape, Tensor):
            if isinstance(max_shape, (list, tuple)) and \
                    len(max_shape) == 2:
                max_shape = proposals.new_tensor(max_shape)
        if isinstance(max_shape, Tensor):
            if max_shape.numel() == 2:
                max_height, max_width = max_shape[0], max_shape[1]
                center_x = center_x.clamp(0, max_width)
                center_y = center_y.clamp(0, max_height)

        decoded = torch.stack([center_x, center_y, width, height, angle],
                              dim=-1)
        if self.use_box_type:
            from mmdet.structures.bbox import RotatedBoxes
            return RotatedBoxes(decoded)
        return decoded


__all__ = ['DeltaXYWHTRBBoxCoder']
