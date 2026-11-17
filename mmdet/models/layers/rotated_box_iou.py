# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

import numpy as np
import torch
from torch import Tensor


def _as_angle_unit(angle_unit: str) -> float:
    """Return the multiplier that converts angles to radians."""
    if angle_unit == 'radian':
        return 1.0
    if angle_unit == 'degree':
        return math.pi / 180.0
    raise ValueError("angle_unit must be either 'radian' or 'degree', "
                     f'but got {angle_unit!r}.')


def _validate_rotated_boxes(boxes: Tensor, name: str) -> None:
    if not isinstance(boxes, Tensor):
        raise TypeError(
            f'{name} must be a torch.Tensor, but got {type(boxes).__name__}.')
    if boxes.dim() < 2 or boxes.size(-1) != 5:
        raise ValueError(
            f'{name} must have shape (..., N, 5) in '
            f'(cx, cy, w, h, angle) format, but got {tuple(boxes.shape)}.')
    if not boxes.is_floating_point():
        raise TypeError(
            f'{name} must be a floating point tensor, but got {boxes.dtype}.')
    if boxes.device.type == 'cpu' and boxes.dtype != torch.bfloat16:
        boxes_np = boxes.detach().numpy()
        non_finite = np.argwhere(~np.isfinite(boxes_np))
        invalid_dim = np.argwhere(boxes_np[..., 2:4] < 0)
    else:
        non_finite = torch.nonzero(~torch.isfinite(boxes), as_tuple=False)
        invalid_dim = torch.nonzero(boxes[..., 2:4] < 0, as_tuple=False)
    if len(non_finite) > 0:
        bad_index = non_finite[0].tolist()
        raise ValueError(
            f'{name} contains a non-finite value at index {bad_index}. '
            f'Input shape: {tuple(boxes.shape)}, dtype: {boxes.dtype}, '
            f'device: {boxes.device}.')
    if len(invalid_dim) > 0:
        bad_index = invalid_dim[0].tolist()
        raise ValueError(
            f'{name} contains a negative width or height at index '
            f'{bad_index}. Empty boxes must use zero width/height. '
            f'Input shape: {tuple(boxes.shape)}, '
            f'device: {boxes.device}.')


def normalize_angle(angles: Tensor,
                    version: str = 'le90',
                    angle_unit: str = 'radian') -> Tensor:
    """Wrap angles into a standard interval without changing box geometry.

    Args:
        angles (Tensor): Angles in radians or degrees.
        version (str): Angle convention. Supported values are ``'le90'``
            ([-pi/2, pi/2)), ``'le135'`` ([-pi/4, 3*pi/4)), ``'r360'``
            ([-pi, pi)), and ``'oc'``. ``'oc'`` leaves the value unchanged;
            call :func:`normalize_rotated_boxes` to wrap and swap edges.
        angle_unit (str): ``'radian'`` or ``'degree'``.

    Returns:
        Tensor: Normalized angles in the same unit as the input.
    """
    if not isinstance(angles, Tensor):
        raise TypeError('angles must be a torch.Tensor, '
                        f'but got {type(angles).__name__}.')
    if not angles.is_floating_point():
        raise TypeError('angles must be floating point, '
                        f'but got {angles.dtype}.')

    multiplier = _as_angle_unit(angle_unit)
    radians = angles * multiplier
    if version == 'le90':
        normalized = torch.remainder(radians + math.pi / 2, math.pi) - \
            math.pi / 2
    elif version == 'le135':
        normalized = torch.remainder(radians + math.pi / 4, math.pi) - \
            math.pi / 4
    elif version == 'r360':
        normalized = torch.remainder(radians + math.pi, 2 * math.pi) - math.pi
    elif version == 'oc':
        normalized = radians
    else:
        raise ValueError(
            "version must be one of 'oc', 'le90', 'le135' or 'r360', "
            f'but got {version!r}.')

    if angle_unit == 'degree':
        normalized = normalized / multiplier
    return normalized


def normalize_rotated_boxes(boxes: Tensor,
                            version: str = 'le90',
                            angle_unit: str = 'radian',
                            width_longer: Optional[bool] = None) -> Tensor:
    """Return a canonical representation of rotated boxes.

    The returned boxes have non-negative dimensions and an angle in the
    interval requested by ``version``. For ``'le90'`` and ``'le135'`` the
    longer edge is stored in ``w``; for ``'oc'`` the shorter edge is stored
    in ``w`` and the angle is in [-pi/2, 0).

    Args:
        boxes (Tensor): Boxes with shape (..., N, 5).
        version (str): Angle convention, see :func:`normalize_angle`.
        angle_unit (str): ``'radian'`` or ``'degree'``.
        width_longer (bool, Optional): Override the edge-swapping rule.

    Returns:
        Tensor: A normalized copy of ``boxes``. Output angles use the same
        unit as the input.
    """
    _validate_rotated_boxes(boxes, 'boxes')
    multiplier = _as_angle_unit(angle_unit)
    angle = boxes[..., 4] * multiplier
    width = boxes[..., 2]
    height = boxes[..., 3]

    if width_longer is None:
        width_longer = version in ('le90', 'le135')
    if version == 'r360':
        normalized_angle = torch.remainder(angle + math.pi, 2 * math.pi) - \
            math.pi
        if angle_unit == 'degree':
            normalized_angle = normalized_angle / multiplier
        return torch.stack(
            [boxes[..., 0], boxes[..., 1], width, height, normalized_angle],
            dim=-1)
    if version == 'oc':
        start_angle = -math.pi / 2
        interval = math.pi
    elif version == 'le135':
        start_angle = -math.pi / 4
        interval = math.pi
    elif version == 'le90':
        start_angle = -math.pi / 2
        interval = math.pi
    else:
        raise ValueError(
            "version must be one of 'oc', 'le90', 'le135' or 'r360', "
            f'but got {version!r}.')

    normalized_angle = torch.remainder(angle - start_angle, interval) + \
        start_angle
    if width_longer:
        swap = width < height
        new_width = torch.where(swap, height, width)
        new_height = torch.where(swap, width, height)
        normalized_angle = torch.where(swap, normalized_angle + math.pi / 2,
                                       normalized_angle)
        normalized_angle = torch.remainder(normalized_angle - start_angle,
                                           interval) + start_angle
    else:
        swap = normalized_angle >= start_angle + interval / 2
        new_width = torch.where(swap, height, width)
        new_height = torch.where(swap, width, height)
        normalized_angle = torch.where(swap, normalized_angle - interval / 2,
                                       normalized_angle)

    if angle_unit == 'degree':
        normalized_angle = normalized_angle / multiplier
    return torch.stack([
        boxes[..., 0], boxes[..., 1], new_width, new_height, normalized_angle
    ],
                       dim=-1)


def rotated_box_to_corners(boxes: Tensor,
                           angle_unit: str = 'radian') -> Tensor:
    """Convert (cx, cy, w, h, angle) boxes to four corners.

    The corner order is ``(-w/2, -h/2)``, ``(w/2, -h/2)``,
    ``(w/2, h/2)``, ``(-w/2, h/2)`` in the box-local frame.

    Args:
        boxes (Tensor): Boxes with shape (..., N, 5).
        angle_unit (str): ``'radian'`` or ``'degree'``.

    Returns:
        Tensor: Corners with shape (..., N, 4, 2).
    """
    _validate_rotated_boxes(boxes, 'boxes')
    angle = boxes[..., 4:5] * _as_angle_unit(angle_unit)
    angle = torch.remainder(angle + math.pi / 2, math.pi) - math.pi / 2
    cos_value = torch.cos(angle)
    sin_value = torch.sin(angle)
    half_w = boxes[..., 2:3] / 2
    half_h = boxes[..., 3:4] / 2
    vec_w = torch.cat([half_w * cos_value, half_w * sin_value], dim=-1)
    vec_h = torch.cat([-half_h * sin_value, half_h * cos_value], dim=-1)
    corners = torch.stack(
        [-vec_w - vec_h, vec_w - vec_h, vec_w + vec_h, -vec_w + vec_h], dim=-2)
    return boxes[..., :2].unsqueeze(-2) + corners


def _clip_polygon(polygons: Tensor, counts: Tensor, values: Tensor,
                  keep_less: bool, tolerance: Tensor) -> tuple:
    """Clip convex polygons against one half-plane.

    ``polygons`` is padded to a fixed number of vertices. ``counts`` records
    the valid prefix for every pair, which lets the same code path run on CPU
    and CUDA without host synchronization.
    """
    num_vertices = polygons.size(-2)
    capacity = num_vertices + 1
    candidate_points = []
    candidate_valid = []

    for vertex_idx in range(num_vertices):
        point = polygons[..., vertex_idx, :]
        next_idx = ((vertex_idx + 1) % counts.clamp(min=1)).unsqueeze(-1)
        gather_index = next_idx.unsqueeze(-1).expand(*next_idx.shape, 2)
        next_point = torch.gather(polygons, -2, gather_index).squeeze(-2)
        current_value = values[..., vertex_idx]
        next_value = torch.gather(values, -1, next_idx).squeeze(-1)

        if keep_less:
            point_inside = current_value <= tolerance
            next_inside = next_value <= tolerance
        else:
            point_inside = current_value >= -tolerance
            next_inside = next_value >= -tolerance

        vertex_valid = (vertex_idx < counts) & point_inside
        crossing = (vertex_idx < counts) & (counts > 0) & \
            (point_inside != next_inside)

        denominator = current_value - next_value
        valid_denominator = denominator.abs() > 1e-12
        safe_denominator = torch.where(valid_denominator, denominator,
                                       torch.ones_like(denominator))
        ratio = torch.where(valid_denominator,
                            current_value / safe_denominator,
                            torch.zeros_like(current_value))
        ratio = ratio.clamp(0, 1)
        intersection = point + (next_point - point) * ratio.unsqueeze(-1)

        candidate_points.extend([point, intersection])
        candidate_valid.extend([vertex_valid, crossing])

    points = torch.stack(candidate_points, dim=-2)
    valid = torch.stack(candidate_valid, dim=-1)
    new_counts = valid.sum(dim=-1)

    slots = torch.cumsum(valid.to(torch.long), dim=-1) - 1
    slots = torch.where(valid, slots, counts.new_full((), capacity))
    output = polygons.new_zeros((*polygons.shape[:-2], capacity + 1, 2))
    scatter_index = slots.unsqueeze(-1).expand(*slots.shape, 2)
    output.scatter_(-2, scatter_index, points)
    return output[..., :capacity, :], new_counts


def _polygon_areas(polygons: Tensor, counts: Tensor) -> Tensor:
    """Compute padded convex-polygon areas."""
    num_vertices = polygons.size(-2)
    vertex_index = torch.arange(
        num_vertices,
        device=polygons.device).view(*((1, ) * (polygons.dim() - 2)),
                                     num_vertices)
    next_index = (vertex_index + 1) % counts.clamp(min=1).unsqueeze(-1)
    next_polygons = torch.gather(
        polygons, -2,
        next_index.unsqueeze(-1).expand(*next_index.shape, 2)).squeeze(-2)
    cross = polygons[..., 0] * next_polygons[..., 1] - \
        polygons[..., 1] * next_polygons[..., 0]
    cross = torch.where(vertex_index < counts.unsqueeze(-1), cross, 0)
    areas = cross.sum(dim=-1).abs() / 2
    return torch.where(counts >= 3, areas, torch.zeros_like(areas))


def _rotated_intersection(boxes1: Tensor,
                          boxes2: Tensor,
                          angle_unit: str = 'radian') -> Tensor:
    """Compute pairwise intersection areas for broadcast batch inputs."""
    corners1 = rotated_box_to_corners(boxes1, angle_unit)
    centers2 = boxes2[..., :2]
    angles2 = boxes2[..., 4] * _as_angle_unit(angle_unit)
    angles2 = torch.remainder(angles2 + math.pi / 2, math.pi) - math.pi / 2

    relative = corners1[..., :, None, :, :] - \
        centers2[..., None, :, None, :]
    # relative: (..., N, M, 4, 2)
    cos_value = torch.cos(-angles2)[..., None, :, None]
    sin_value = torch.sin(-angles2)[..., None, :, None]
    local_x = relative[..., 0] * cos_value - relative[..., 1] * sin_value
    local_y = relative[..., 0] * sin_value + relative[..., 1] * cos_value
    polygons = torch.stack([local_x, local_y], dim=-1)
    counts = torch.full(
        polygons.shape[:-2], 4, dtype=torch.long, device=polygons.device)

    half_w = (boxes2[..., 2] / 2)[..., None, :, None]
    half_h = (boxes2[..., 3] / 2)[..., None, :, None]
    scale = torch.maximum(boxes1[..., 2:4].amax(dim=-1)[..., :, None],
                          boxes2[..., 2:4].amax(dim=-1)[..., None, :])
    tolerance = scale.clamp_min(1) * 1e-7
    clip_planes = (
        (0, half_w, True),
        (1, half_h, True),
        (0, -half_w, False),
        (1, -half_h, False),
    )
    for axis, bound, keep_less in clip_planes:
        values = polygons[..., axis] - bound
        polygons, counts = _clip_polygon(polygons, counts, values, keep_less,
                                         tolerance)

    intersection = _polygon_areas(polygons, counts)
    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    max_intersection = torch.minimum(area1[..., :, None], area2[..., None, :])
    intersection = intersection.clamp(min=0)
    intersection = torch.minimum(intersection, max_intersection)
    return torch.where((area1[..., :, None] > 0) & (area2[..., None, :] > 0),
                       intersection, torch.zeros_like(intersection))


def box_iou_rotated(boxes1: Tensor,
                    boxes2: Tensor,
                    mode: str = 'iou',
                    aligned: bool = False,
                    clockwise: bool = True,
                    angle_unit: str = 'radian',
                    eps: float = 1e-6) -> Tensor:
    """Calculate IoU or IoF between two sets of rotated boxes.

    Box format is ``(cx, cy, w, h, angle)``. ``angle`` is measured in radians
    by default and follows the MMCV convention (positive is clockwise in an
    image coordinate system whose positive y-axis points down).

    Args:
        boxes1 (Tensor): Boxes with shape (..., N, 5).
        boxes2 (Tensor): Boxes with shape (..., M, 5).
        mode (str): ``'iou'`` or ``'iof'``.
        aligned (bool): If ``True``, calculate only the diagonal IoUs. The
            number of boxes in both inputs must match.
        clockwise (bool): Set to ``False`` when angles use the
            counter-clockwise convention.
        angle_unit (str): ``'radian'`` or ``'degree'``.
        eps (float): Positive denominator clamp for numerical stability.

    Returns:
        Tensor: Shape (..., N, M), or (..., N) when ``aligned=True``.

    Raises:
        ValueError: If shapes, angles, dimensions or values are invalid.
        TypeError: If inputs are not floating point tensors.
    """
    _validate_rotated_boxes(boxes1, 'boxes1')
    _validate_rotated_boxes(boxes2, 'boxes2')
    if boxes1.device != boxes2.device:
        raise ValueError(
            f'box_iou_rotated requires boxes on the same device, but got '
            f'{boxes1.device} and {boxes2.device}.')
    if boxes1.dtype != boxes2.dtype:
        raise ValueError(
            f'box_iou_rotated requires boxes with the same dtype, but got '
            f'{boxes1.dtype} and {boxes2.dtype}.')
    if mode not in ('iou', 'iof'):
        raise ValueError(
            f"mode must be either 'iou' or 'iof', but got {mode!r}.")
    if eps <= 0:
        raise ValueError(f'eps must be positive, but got {eps}.')
    _as_angle_unit(angle_unit)

    if not clockwise:
        boxes1 = boxes1.clone()
        boxes2 = boxes2.clone()
        boxes1[..., 4] = -boxes1[..., 4]
        boxes2[..., 4] = -boxes2[..., 4]

    try:
        batch_shape = torch.broadcast_shapes(boxes1.shape[:-2],
                                             boxes2.shape[:-2])
    except RuntimeError as error:
        raise ValueError('The batch dimensions of boxes1 and boxes2 are not '
                         f'broadcastable: {tuple(boxes1.shape)} and '
                         f'{tuple(boxes2.shape)}.') from error

    boxes1 = boxes1.expand(*batch_shape, *boxes1.shape[-2:])
    boxes2 = boxes2.expand(*batch_shape, *boxes2.shape[-2:])
    rows, cols = boxes1.size(-2), boxes2.size(-2)
    if aligned and rows != cols:
        raise ValueError(
            'aligned=True requires the same number of boxes, but got '
            f'{rows} and {cols}.')
    if rows == 0 or cols == 0:
        shape = (*batch_shape, rows) if aligned else (*batch_shape, rows, cols)
        return boxes1.new_zeros(shape)

    if aligned:
        intersection = _rotated_intersection(
            boxes1.unsqueeze(-2), boxes2.unsqueeze(-2),
            angle_unit).squeeze(-1).squeeze(-1)
    else:
        intersection = _rotated_intersection(boxes1, boxes2, angle_unit)

    area1 = boxes1[..., 2] * boxes1[..., 3]
    area2 = boxes2[..., 2] * boxes2[..., 3]
    if aligned:
        if mode == 'iou':
            denominator = area1 + area2 - intersection
        else:
            denominator = area1
        union = denominator.clamp_min(eps)
        ious = intersection / union
    else:
        if mode == 'iou':
            denominator = area1[..., :, None] + area2[..., None, :] - \
                intersection
        else:
            denominator = area1[..., :, None].expand_as(intersection)
        union = denominator.clamp_min(eps)
        ious = intersection / union
    return ious.clamp(0, 1)


def rbox_overlaps(boxes1: Tensor,
                  boxes2: Tensor,
                  mode: str = 'iou',
                  aligned: bool = False,
                  **kwargs) -> Tensor:
    """Alias of :func:`box_iou_rotated` used by MMDetection components."""
    return box_iou_rotated(
        boxes1, boxes2, mode=mode, aligned=aligned, **kwargs)


def rotated_box_iou(boxes1: Tensor,
                    boxes2: Tensor,
                    mode: str = 'iou',
                    is_aligned: bool = False,
                    **kwargs) -> Tensor:
    """Calculate rotated-box overlaps with an ``is_aligned`` keyword."""
    return box_iou_rotated(
        boxes1, boxes2, mode=mode, aligned=is_aligned, **kwargs)


def rbbox_overlaps(boxes1: Tensor,
                   boxes2: Tensor,
                   mode: str = 'iou',
                   is_aligned: bool = False,
                   **kwargs) -> Tensor:
    """MMRotate-compatible alias of :func:`box_iou_rotated`."""
    return box_iou_rotated(
        boxes1, boxes2, mode=mode, aligned=is_aligned, **kwargs)


norm_angle = normalize_angle

__all__ = [
    'box_iou_rotated', 'rbox_overlaps', 'rbbox_overlaps', 'rotated_box_iou',
    'rotated_box_to_corners', 'normalize_angle', 'normalize_rotated_boxes',
    'norm_angle'
]
