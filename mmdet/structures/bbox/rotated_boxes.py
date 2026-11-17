# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch import BoolTensor, Tensor

from mmdet.structures.mask.structures import BitmapMasks, PolygonMasks
from .base_boxes import BaseBoxes
from .box_type import register_box, register_box_converter

DeviceType = Union[str, torch.device]
MaskType = Union[BitmapMasks, PolygonMasks]


@register_box('rbox')
class RotatedBoxes(BaseBoxes):
    """Rotated boxes in ``(cx, cy, w, h, angle)`` format.

    The angle is stored in radians and follows the OpenCV/MMCV convention:
    positive angles rotate clockwise in an image coordinate system where the
    y-axis points down.

    Args:
        data (Tensor, ndarray or Sequence): Box data with shape (..., 5).
        dtype (torch.dtype, optional): Output dtype.
        device (str or torch.device, optional): Output device.
        clone (bool): Whether to clone the input data. Defaults to True.
    """

    box_dim = 5

    def regularize_boxes(self,
                         pattern: Optional[str] = None,
                         width_longer: bool = True,
                         start_angle: float = -90) -> Tensor:
        """Regularize box angles and optionally swap width and height.

        Supported presets are ``'oc'``, ``'le90'`` and ``'le135'``. The
        returned angle is always represented in radians.
        """
        if pattern is not None:
            if pattern == 'oc':
                width_longer, start_angle = False, -90
            elif pattern == 'le90':
                width_longer, start_angle = True, -90
            elif pattern == 'le135':
                width_longer, start_angle = True, -45
            else:
                raise ValueError(
                    "pattern must be one of 'oc', 'le90' or 'le135', "
                    f'but got {pattern!r}.')

        start_angle_rad = start_angle / 180 * np.pi
        x, y, width, height, angle = self.tensor.unbind(dim=-1)
        if width_longer:
            swap = height > width
            new_width = torch.where(swap, height, width)
            new_height = torch.where(swap, width, height)
            angle = torch.where(swap, angle + np.pi / 2, angle)
            angle = torch.remainder(angle - start_angle_rad, np.pi) + \
                start_angle_rad
        else:
            angle = torch.remainder(angle - start_angle_rad, np.pi)
            swap = angle >= np.pi / 2
            new_width = torch.where(swap, height, width)
            new_height = torch.where(swap, width, height)
            angle = torch.where(swap, angle - np.pi / 2, angle) + \
                start_angle_rad
        self.tensor = torch.stack([x, y, new_width, new_height, angle], dim=-1)
        return self.tensor

    @property
    def centers(self) -> Tensor:
        return self.tensor[..., :2]

    @property
    def areas(self) -> Tensor:
        return self.tensor[..., 2] * self.tensor[..., 3]

    @property
    def widths(self) -> Tensor:
        return self.tensor[..., 2]

    @property
    def heights(self) -> Tensor:
        return self.tensor[..., 3]

    def flip_(self,
              img_shape: Tuple[int, int],
              direction: str = 'horizontal') -> None:
        boxes = self.tensor
        if direction == 'horizontal':
            boxes[..., 0] = img_shape[1] - boxes[..., 0]
            boxes[..., 4] = -boxes[..., 4]
        elif direction == 'vertical':
            boxes[..., 1] = img_shape[0] - boxes[..., 1]
            boxes[..., 4] = -boxes[..., 4]
        elif direction == 'diagonal':
            boxes[..., 0] = img_shape[1] - boxes[..., 0]
            boxes[..., 1] = img_shape[0] - boxes[..., 1]
        else:
            raise ValueError(
                "direction must be 'horizontal', 'vertical' or 'diagonal', "
                f'but got {direction!r}.')

    def translate_(self, distances: Tuple[float, float]) -> None:
        if len(distances) != 2:
            raise ValueError(
                f'distances must contain two values, but got {distances}.')
        self.tensor[..., :2] += self.tensor.new_tensor(distances)

    def clip_(self, img_shape: Tuple[int, int]) -> None:
        """Rotated boxes are not clipped in-place.

        Clipping an oriented rectangle to an image boundary generally changes
        it into a polygon, so this method intentionally leaves the data
        unchanged.
        """
        warnings.warn(
            'RotatedBoxes.clip_ does not modify boxes because clipping a '
            'rotated rectangle can produce a general polygon. '
            f'Image shape {img_shape} and boxes {self.tensor.shape} were '
            'left unchanged.',
            stacklevel=2)

    def rotate_(self, center: Tuple[float, float], angle: float) -> None:
        rotation_matrix = self.tensor.new_tensor(
            cv2.getRotationMatrix2D(center, -angle, 1))
        centers, wh, angles = torch.split(self.tensor, [2, 2, 1], dim=-1)
        angles = angles + angle / 180 * np.pi
        homogeneous = torch.cat(
            [centers, centers.new_ones(*centers.shape[:-1], 1)], dim=-1)
        rotated = torch.matmul(rotation_matrix,
                               homogeneous.transpose(-1,
                                                     -2)).transpose(-1, -2)
        self.tensor = torch.cat([rotated, wh, angles], dim=-1)

    def project_(self, homography_matrix: Union[Tensor, np.ndarray]) -> None:
        if isinstance(homography_matrix, np.ndarray):
            homography_matrix = self.tensor.new_tensor(homography_matrix)
        corners = self.rbox2corner(self.tensor)
        corners = torch.cat(
            [corners, corners.new_ones(*corners.shape[:-1], 1)], dim=-1)
        corners = torch.matmul(homography_matrix,
                               corners.transpose(-1, -2)).transpose(-1, -2)
        corners = corners[..., :2] / corners[..., 2:3]
        self.tensor = self.corner2rbox(corners)

    @staticmethod
    def rbox2corner(boxes: Tensor) -> Tensor:
        """Convert ``(cx, cy, w, h, angle)`` boxes to four corners."""
        centers, width, height, angle = torch.split(
            boxes, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(angle), torch.sin(angle)
        vec_w = torch.cat([width / 2 * cos_value, width / 2 * sin_value],
                          dim=-1)
        vec_h = torch.cat([-height / 2 * sin_value, height / 2 * cos_value],
                          dim=-1)
        corners = torch.stack([
            centers + vec_w + vec_h, centers + vec_w - vec_h,
            centers - vec_w - vec_h, centers - vec_w + vec_h
        ],
                              dim=-2)
        return corners

    @staticmethod
    def corner2rbox(corners: Tensor) -> Tensor:
        """Convert corner points to minimum-area rotated boxes using OpenCV."""
        original_shape = corners.shape[:-2]
        points = corners.detach().cpu().numpy().reshape(-1, 4, 2)
        if points.shape[0] == 0:
            return corners.new_zeros(*original_shape, 5)
        boxes = []
        for point_set in points:
            try:
                center, size, angle = cv2.minAreaRect(
                    point_set.astype(np.float32))
            except cv2.error:
                boxes.append([0., 0., 0., 0., 0.])
            else:
                boxes.append([
                    center[0], center[1], size[0], size[1], angle / 180 * np.pi
                ])
        return corners.new_tensor(boxes).reshape(*original_shape, 5)

    def rescale_(self, scale_factor: Tuple[float, float]) -> None:
        if len(scale_factor) != 2:
            raise ValueError('scale_factor must contain two values, but got '
                             f'{scale_factor}.')
        scale_x, scale_y = scale_factor
        centers, width, height, angle = torch.split(
            self.tensor, [2, 1, 1, 1], dim=-1)
        cos_value, sin_value = torch.cos(angle), torch.sin(angle)
        centers = centers * centers.new_tensor([scale_x, scale_y])
        width = width * torch.sqrt((scale_x * cos_value).square() +
                                   (scale_y * sin_value).square())
        height = height * torch.sqrt((scale_x * sin_value).square() +
                                     (scale_y * cos_value).square())
        angle = torch.atan2(scale_x * sin_value, scale_y * cos_value)
        self.tensor = torch.cat([centers, width, height, angle], dim=-1)

    def resize_(self, scale_factor: Tuple[float, float]) -> None:
        if len(scale_factor) != 2:
            raise ValueError('scale_factor must contain two values, but got '
                             f'{scale_factor}.')
        centers, wh, angle = torch.split(self.tensor, [2, 2, 1], dim=-1)
        wh = wh * self.tensor.new_tensor(scale_factor)
        self.tensor = torch.cat([centers, wh, angle], dim=-1)

    def is_inside(self,
                  img_shape: Tuple[int, int],
                  all_inside: bool = False,
                  allowed_border: int = 0) -> BoolTensor:
        del all_inside
        boxes = self.tensor
        return (boxes[..., 0] >= -allowed_border) & \
            (boxes[..., 1] >= -allowed_border) & \
            (boxes[..., 0] <= img_shape[1] + allowed_border) & \
            (boxes[..., 1] <= img_shape[0] + allowed_border)

    def find_inside_points(self,
                           points: Tensor,
                           is_aligned: bool = False,
                           eps: float = 0.01) -> BoolTensor:
        boxes = self.tensor
        if boxes.dim() != 2:
            raise ValueError(
                'find_inside_points requires 2D boxes, but got shape '
                f'{tuple(boxes.shape)}.')
        if not is_aligned:
            boxes = boxes[None]
            points = points[:, None]
        elif boxes.size(0) != points.size(0):
            raise ValueError(
                'is_aligned=True requires the same number of boxes and '
                f'points, but got {boxes.size(0)} and {points.size(0)}.')

        centers, wh, angle = torch.split(boxes, [2, 2, 1], dim=-1)
        cos_value, sin_value = torch.cos(angle), torch.sin(angle)
        matrix = torch.cat([cos_value, sin_value, -sin_value, cos_value],
                           dim=-1).reshape(*boxes.shape[:-1], 2, 2)
        offset = torch.matmul(matrix,
                              (points - centers).unsqueeze(-1)).squeeze(-1)
        width, height = wh[..., 0], wh[..., 1]
        return (offset[..., 0] <= width / 2 - eps) & \
            (offset[..., 0] >= -width / 2 + eps) & \
            (offset[..., 1] <= height / 2 - eps) & \
            (offset[..., 1] >= -height / 2 + eps)

    @staticmethod
    def overlaps(boxes1: BaseBoxes,
                 boxes2: BaseBoxes,
                 mode: str = 'iou',
                 is_aligned: bool = False,
                 eps: float = 1e-6) -> Tensor:
        from mmdet.models.layers.rotated_box_iou import box_iou_rotated
        if not isinstance(boxes1, BaseBoxes):
            boxes1 = RotatedBoxes(boxes1)
        if not isinstance(boxes2, BaseBoxes):
            boxes2 = RotatedBoxes(boxes2)
        boxes1 = boxes1.convert_to('rbox')
        boxes2 = boxes2.convert_to('rbox')
        return box_iou_rotated(
            boxes1.tensor,
            boxes2.tensor,
            mode=mode,
            aligned=is_aligned,
            eps=eps)

    @staticmethod
    def from_instance_masks(masks: MaskType) -> 'RotatedBoxes':
        if not isinstance(masks, (BitmapMasks, PolygonMasks)):
            raise TypeError(
                'masks must be BitmapMasks or PolygonMasks, but got '
                f'{type(masks).__name__}.')
        boxes = []
        if isinstance(masks, BitmapMasks):
            for mask in masks.masks:
                points = np.stack(np.nonzero(mask), axis=-1).astype(np.float32)
                if points.shape[0] < 3:
                    boxes.append([0., 0., 0., 0., 0.])
                else:
                    center, size, angle = cv2.minAreaRect(points)
                    boxes.append([
                        center[0], center[1], size[0], size[1],
                        angle / 180 * np.pi
                    ])
        else:
            for polygons in masks.masks:
                if len(polygons) == 0:
                    boxes.append([0., 0., 0., 0., 0.])
                    continue
                points = np.concatenate(
                    [np.asarray(poly).reshape(-1, 2) for poly in polygons],
                    axis=0).astype(np.float32)
                if points.shape[0] < 3:
                    boxes.append([0., 0., 0., 0., 0.])
                else:
                    center, size, angle = cv2.minAreaRect(points)
                    boxes.append([
                        center[0], center[1], size[0], size[1],
                        angle / 180 * np.pi
                    ])
        return RotatedBoxes(boxes, dtype=torch.float32)


def hbox2rbox(boxes: Tensor) -> Tensor:
    """Convert horizontal ``(x1, y1, x2, y2)`` boxes to rotated boxes."""
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1,
                        boxes.new_zeros(x1.shape)],
                       dim=-1)


def rbox2hbox(boxes: Tensor) -> Tensor:
    """Convert rotated boxes to their axis-aligned bounding boxes."""
    corners = RotatedBoxes.rbox2corner(boxes)
    x_min = corners[..., 0].min(dim=-1).values
    y_min = corners[..., 1].min(dim=-1).values
    x_max = corners[..., 0].max(dim=-1).values
    y_max = corners[..., 1].max(dim=-1).values
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


register_box_converter('hbox', 'rbox', hbox2rbox)
register_box_converter('rbox', 'hbox', rbox2hbox)

__all__ = ['RotatedBoxes', 'hbox2rbox', 'rbox2hbox']
