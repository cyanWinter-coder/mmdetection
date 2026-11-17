# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .rotated_box_iou import box_iou_rotated

_MAX_EXACT_EDGES = 1_500_000
_EXACT_IOU_BATCH = 131_072


class _CandidateLimitExceeded(RuntimeError):
    """Internal signal used to fall back to sequential greedy NMS."""


def _validate_nms_inputs(dets: Tensor, scores: Tensor,
                         labels: Optional[Tensor], iou_threshold: float,
                         angle_unit: str) -> None:
    if dets.dim() != 2 or dets.size(-1) != 5:
        raise ValueError('nms_rotated expects dets with shape (N, 5) in '
                         '(cx, cy, w, h, angle) format, '
                         f'but got {tuple(dets.shape)}.')
    if not dets.is_floating_point():
        raise TypeError(f'dets must be floating point, but got {dets.dtype}.')
    if scores.dim() != 1 or scores.size(0) != dets.size(0):
        raise ValueError('scores must have shape (N,) matching dets, but got '
                         f'{tuple(scores.shape)} and {tuple(dets.shape)}.')
    if scores.device != dets.device:
        raise ValueError('dets and scores must be on the same device, but got '
                         f'{dets.device} and {scores.device}.')
    if scores.dtype != dets.dtype:
        raise ValueError('dets and scores must have the same dtype, but got '
                         f'{dets.dtype} and {scores.dtype}.')
    use_numpy = dets.device.type == 'cpu' and dets.dtype != torch.bfloat16
    if use_numpy:
        dets_np = dets.detach().numpy()
        scores_np = scores.detach().numpy()
        non_finite_dets = np.argwhere(~np.isfinite(dets_np))
        non_finite_scores = np.argwhere(~np.isfinite(scores_np))
        invalid_dim = np.argwhere(dets_np[:, 2:4] < 0)
    else:
        non_finite_dets = torch.nonzero(~torch.isfinite(dets), as_tuple=False)
        non_finite_scores = torch.nonzero(
            ~torch.isfinite(scores), as_tuple=False)
        invalid_dim = torch.nonzero(dets[:, 2:4] < 0, as_tuple=False)
    if len(non_finite_scores) > 0:
        bad_index = non_finite_scores[0].tolist()
        raise ValueError('nms_rotated received a non-finite score at index '
                         f'{bad_index}. Input shape: {tuple(scores.shape)}, '
                         f'device: {scores.device}.')
    if len(non_finite_dets) > 0:
        bad_index = non_finite_dets[0].tolist()
        raise ValueError(
            'nms_rotated received a non-finite box value at index '
            f'{bad_index}. Input shape: {tuple(dets.shape)}, '
            f'device: {dets.device}.')
    if len(invalid_dim) > 0:
        bad_index = invalid_dim[0].tolist()
        raise ValueError(
            'nms_rotated received a negative width or height at index '
            f'{bad_index}. Empty boxes must use zero width/height.')
    if labels is not None:
        if labels.dim() != 1 or labels.size(0) != dets.size(0):
            raise ValueError(
                'labels must have shape (N,) matching dets, but got '
                f'{tuple(labels.shape)} and {tuple(dets.shape)}.')
        if labels.device != dets.device:
            raise ValueError(
                'dets and labels must be on the same device, but got '
                f'{dets.device} and {labels.device}.')
        if labels.is_floating_point():
            if use_numpy:
                non_finite_labels = np.argwhere(
                    ~np.isfinite(labels.detach().numpy()))
            else:
                non_finite_labels = torch.nonzero(
                    ~torch.isfinite(labels), as_tuple=False)
            if len(non_finite_labels) > 0:
                bad_index = non_finite_labels[0].tolist()
                raise ValueError(
                    'nms_rotated received a non-finite label at index '
                    f'{bad_index}. Input shape: {tuple(labels.shape)}, '
                    f'device: {labels.device}.')
    if not 0 <= iou_threshold <= 1:
        raise ValueError(
            f'iou_threshold must be in [0, 1], but got {iou_threshold}.')
    if angle_unit not in ('radian', 'degree'):
        raise ValueError("angle_unit must be either 'radian' or 'degree', "
                         f'but got {angle_unit!r}.')


def _box_half_extents(boxes: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    width, height = boxes[..., 2], boxes[..., 3]
    angle = torch.remainder(boxes[..., 4] + math.pi / 2, math.pi) - math.pi / 2
    cos_value = torch.cos(angle).abs()
    sin_value = torch.sin(angle).abs()
    extent_x = (width * cos_value + height * sin_value) / 2
    extent_y = (width * sin_value + height * cos_value) / 2
    radius = torch.sqrt(width.square() + height.square()) / 2
    return extent_x, extent_y, radius


def _exact_overlap_pairs(boxes: Tensor, row_index: Tensor, col_index: Tensor,
                         iou_threshold: float, angle_unit: str,
                         clockwise: bool, keep_rows: list,
                         keep_cols: list) -> None:
    for start in range(0, row_index.numel(), _EXACT_IOU_BATCH):
        end = start + _EXACT_IOU_BATCH
        rows = row_index[start:end]
        cols = col_index[start:end]
        overlaps = box_iou_rotated(
            boxes[rows],
            boxes[cols],
            aligned=True,
            clockwise=clockwise,
            angle_unit=angle_unit)
        valid = overlaps >= iou_threshold
        if valid.any():
            keep_rows.extend(rows[valid].tolist())
            keep_cols.extend(cols[valid].tolist())


def _sweep_candidate_pairs(boxes: Tensor,
                           labels: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    """Generate AABB-overlapping pairs with a CPU sweep line.

    A pair of oriented rectangles can only overlap when their axis-aligned
    bounding boxes overlap. Sorting by the left edge lets each box inspect
    only intervals that start before its right edge, reducing the broad phase
    from O(N^2) to roughly O(N + K), where K is the number of candidates.
    """
    boxes_cpu = boxes.detach().cpu().numpy()
    centers = boxes_cpu[:, :2]
    width = boxes_cpu[:, 2]
    height = boxes_cpu[:, 3]
    angle = np.remainder(boxes_cpu[:, 4] + np.pi / 2, np.pi) - np.pi / 2
    cos_value = np.abs(np.cos(angle))
    sin_value = np.abs(np.sin(angle))
    extent_x = (width * cos_value + height * sin_value) / 2
    extent_y = (width * sin_value + height * cos_value) / 2
    radius = np.sqrt(np.square(width) + np.square(height)) / 2
    left = centers[:, 0] - extent_x
    right = centers[:, 0] + extent_x
    order = np.argsort(left, kind='stable')
    left_sorted = left[order]
    labels_cpu = labels.detach().cpu().numpy() if labels is not None else None

    sweep_tolerance = 1e-6 * np.maximum(1, np.abs(right[order]))
    ends = np.searchsorted(
        left_sorted, right[order] + sweep_tolerance, side='right')
    starts = np.arange(order.size) + 1
    counts = np.maximum(ends - starts, 0)
    num_x_candidates = int(counts.sum())
    if num_x_candidates > _MAX_EXACT_EDGES * 2:
        raise _CandidateLimitExceeded(
            f'{num_x_candidates} x-axis candidates exceed the safe limit '
            f'{_MAX_EXACT_EDGES * 2}.')
    if num_x_candidates > 0:
        row_positions = np.repeat(np.arange(order.size), counts)
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        candidate_positions = starts[row_positions] + \
            (np.arange(num_x_candidates) - offsets)
        row_index = order[row_positions]
        col_index = order[candidate_positions]

        delta_x = centers[col_index, 0] - centers[row_index, 0]
        delta_y = centers[col_index, 1] - centers[row_index, 1]
        tolerance_x = 1e-6 * np.maximum(
            1, np.maximum(extent_x[col_index], extent_x[row_index]))
        tolerance_y = 1e-6 * np.maximum(
            1, np.maximum(extent_y[col_index], extent_y[row_index]))
        valid = np.abs(delta_x) <= \
            extent_x[col_index] + extent_x[row_index] + tolerance_x
        valid &= np.abs(delta_y) <= \
            extent_y[col_index] + extent_y[row_index] + tolerance_y
        radius_sum = radius[col_index] + radius[row_index]
        valid &= np.square(delta_x) + np.square(delta_y) <= \
            np.square(radius_sum) + 1e-6 * np.maximum(
                1, np.square(radius_sum))
        if labels_cpu is not None:
            valid &= labels_cpu[col_index] == labels_cpu[row_index]
        row_index = row_index[valid]
        col_index = col_index[valid]
    else:
        row_index = np.empty(0, dtype=np.int64)
        col_index = np.empty(0, dtype=np.int64)

    if row_index.size > _MAX_EXACT_EDGES:
        raise _CandidateLimitExceeded(
            f'{row_index.size} broad-phase candidates exceed the safe '
            f'edge limit {_MAX_EXACT_EDGES}.')
    if row_index.size:
        rows = torch.from_numpy(np.minimum(row_index,
                                           col_index)).to(boxes.device)
        cols = torch.from_numpy(np.maximum(row_index,
                                           col_index)).to(boxes.device)
        return rows, cols
    return boxes.new_empty((0, ), dtype=torch.long), boxes.new_empty(
        (0, ), dtype=torch.long)


def _greedy_nms_by_pairs(boxes: Tensor, labels: Optional[Tensor],
                         iou_threshold: float, angle_unit: str,
                         clockwise: bool, max_num: int) -> list:
    num_boxes = boxes.size(0)
    row_index, col_index = _sweep_candidate_pairs(boxes, labels)

    suppress_rows = []
    suppress_cols = []
    _exact_overlap_pairs(boxes, row_index, col_index, iou_threshold,
                         angle_unit, clockwise, suppress_rows, suppress_cols)

    neighbors = [[] for _ in range(num_boxes)]
    for row, col in zip(suppress_rows, suppress_cols):
        neighbors[row].append(col)

    suppressed = bytearray(num_boxes)
    keep = []
    for index in range(num_boxes):
        if suppressed[index]:
            continue
        keep.append(index)
        if max_num > 0 and len(keep) >= max_num:
            break
        for other in neighbors[index]:
            suppressed[other] = 1
    return keep


def _greedy_nms_fallback(boxes: Tensor, labels: Optional[Tensor],
                         iou_threshold: float, angle_unit: str,
                         clockwise: bool, max_num: int) -> list:
    """Sequential greedy NMS for dense candidate graphs."""
    num_boxes = boxes.size(0)
    centers = boxes[:, :2]
    extent_x, extent_y, radius = _box_half_extents(boxes)
    suppressed = torch.zeros(num_boxes, dtype=torch.bool, device=boxes.device)
    keep = []
    for index in range(num_boxes):
        if suppressed[index]:
            continue
        keep.append(index)
        if max_num > 0 and len(keep) >= max_num:
            break
        if index + 1 >= num_boxes:
            continue

        remaining = torch.arange(index + 1, num_boxes, device=boxes.device)
        remaining = remaining[~suppressed[index + 1:]]
        if labels is not None:
            remaining = remaining[labels[remaining] == labels[index]]
        if remaining.numel() == 0:
            continue

        delta_x = centers[remaining, 0] - centers[index, 0]
        delta_y = centers[remaining, 1] - centers[index, 1]
        radius_sum = radius[remaining] + radius[index]
        candidate = delta_x.square() + delta_y.square() <= \
            radius_sum.square() + 1e-6 * radius_sum.square().clamp_min(1)
        extent_sum_x = extent_x[remaining] + extent_x[index]
        extent_sum_y = extent_y[remaining] + extent_y[index]
        candidate &= delta_x.abs() <= \
            extent_sum_x + 1e-6 * extent_sum_x.clamp_min(1)
        candidate &= delta_y.abs() <= \
            extent_sum_y + 1e-6 * extent_sum_y.clamp_min(1)
        remaining = remaining[candidate]
        if remaining.numel() == 0:
            continue
        overlaps = box_iou_rotated(
            boxes[index].unsqueeze(0).expand(remaining.numel(), 5),
            boxes[remaining],
            aligned=True,
            clockwise=clockwise,
            angle_unit=angle_unit)
        suppressed[remaining[overlaps >= iou_threshold]] = True
    return keep


def _nms_rotated_torch(dets: Tensor, scores: Tensor, labels: Optional[Tensor],
                       iou_threshold: float, clockwise: bool, angle_unit: str,
                       max_num: int) -> Tensor:
    _validate_nms_inputs(dets, scores, labels, iou_threshold, angle_unit)
    if dets.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=dets.device)

    if angle_unit == 'degree':
        dets = dets.clone()
        dets[:, 4] = dets[:, 4] * (math.pi / 180)
        angle_unit = 'radian'

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_boxes = dets[order]
    sorted_labels = labels[order] if labels is not None else None
    if iou_threshold == 0:
        # At threshold zero every lower-scoring box is suppressed, including
        # boxes with zero IoU. With labels, keep the first box per label.
        if sorted_labels is None:
            keep_sorted = [0]
        else:
            seen_labels = set()
            keep_sorted = []
            for index, label in enumerate(sorted_labels.tolist()):
                if label not in seen_labels:
                    seen_labels.add(label)
                    keep_sorted.append(index)
                    if max_num > 0 and len(keep_sorted) >= max_num:
                        break
        if max_num > 0:
            keep_sorted = keep_sorted[:max_num]
    elif sorted_boxes.size(0) <= 256:
        keep_sorted = _greedy_nms_fallback(sorted_boxes, sorted_labels,
                                           iou_threshold, angle_unit,
                                           clockwise, max_num)
    else:
        try:
            keep_sorted = _greedy_nms_by_pairs(sorted_boxes, sorted_labels,
                                               iou_threshold, angle_unit,
                                               clockwise, max_num)
        except _CandidateLimitExceeded:
            keep_sorted = _greedy_nms_fallback(sorted_boxes, sorted_labels,
                                               iou_threshold, angle_unit,
                                               clockwise, max_num)
    if not keep_sorted:
        return torch.empty(0, dtype=torch.long, device=dets.device)
    keep_sorted = torch.as_tensor(
        keep_sorted, dtype=torch.long, device=dets.device)
    return order[keep_sorted]


def nms_rotated(
    dets: Union[Tensor, np.ndarray],
    scores: Union[Tensor, np.ndarray],
    iou_threshold: float,
    labels: Optional[Union[Tensor, np.ndarray]] = None,
    clockwise: bool = True,
    angle_unit: str = 'radian',
    max_num: int = -1
) -> Tuple[Union[Tensor, np.ndarray], Union[Tensor, np.ndarray]]:
    """Perform NMS on rotated boxes.

    Boxes use ``(cx, cy, w, h, angle)`` format. Suppression uses
    ``IoU >= iou_threshold`` to match MMCV's rotated NMS boundary behaviour.

    Args:
        dets (Tensor or ndarray): Boxes with shape (N, 5).
        scores (Tensor or ndarray): Scores with shape (N,).
        iou_threshold (float): IoU threshold in [0, 1].
        labels (Tensor or ndarray, optional): If provided, only boxes with
            the same label suppress each other.
        clockwise (bool): Angle direction convention.
        angle_unit (str): ``'radian'`` or ``'degree'``.
        max_num (int): Maximum kept boxes. Use -1 to keep all.

    Returns:
        tuple: Kept boxes with scores (K, 6) and original indices (K,).
    """
    if max_num == 0:
        raise ValueError('max_num must be positive or -1, but got 0.')
    if max_num < -1:
        raise ValueError(f'max_num must be positive or -1, but got {max_num}.')

    is_numpy = isinstance(dets, np.ndarray)
    if is_numpy:
        if not isinstance(scores, np.ndarray):
            raise TypeError(
                'When dets is a numpy array, scores must also be a numpy '
                f'array, but got {type(scores).__name__}.')
        dets_tensor = torch.from_numpy(dets)
        scores_tensor = torch.from_numpy(scores)
        labels_tensor = None if labels is None else torch.from_numpy(labels)
    else:
        if not isinstance(dets, Tensor) or not isinstance(scores, Tensor):
            raise TypeError(
                'dets and scores must both be torch.Tensor or both be '
                'numpy.ndarray.')
        dets_tensor = dets
        scores_tensor = scores
        if labels is not None and not isinstance(labels, Tensor):
            raise TypeError(
                'When dets is a torch.Tensor, labels must also be a '
                f'torch.Tensor, but got {type(labels).__name__}.')
        labels_tensor = labels

    keep = _nms_rotated_torch(dets_tensor, scores_tensor, labels_tensor,
                              iou_threshold, clockwise, angle_unit, max_num)
    kept_dets = torch.cat([dets_tensor[keep], scores_tensor[keep, None]],
                          dim=-1)
    if is_numpy:
        return kept_dets.numpy(), keep.numpy()
    return kept_dets, keep


def batched_nms_rotated(boxes: Tensor,
                        scores: Tensor,
                        idxs: Optional[Tensor],
                        nms_cfg: Optional[dict],
                        class_agnostic: bool = False) -> Tuple[Tensor, Tensor]:
    """Apply rotated NMS independently per label or class-agnostically.

    Args:
        boxes (Tensor): Rotated boxes with shape (N, 5).
        scores (Tensor): Scores with shape (N,).
        idxs (Tensor, optional): Label/group index for each box.
        nms_cfg (dict, optional): NMS config. ``None`` only sorts by score.
        class_agnostic (bool): Suppress across all labels when ``True``.

    Returns:
        tuple[Tensor, Tensor]: Boxes with appended scores and kept indices.
    """
    if scores.dim() != 1 or scores.size(0) != boxes.size(0):
        raise ValueError('scores must have shape (N,) matching boxes, but got '
                         f'{tuple(scores.shape)} and {tuple(boxes.shape)}.')
    if idxs is not None and (idxs.dim() != 1 or idxs.size(0) != boxes.size(0)):
        raise ValueError('idxs must have shape (N,) matching boxes, but got '
                         f'{tuple(idxs.shape)} and {tuple(boxes.shape)}.')

    if nms_cfg is None:
        keep = torch.argsort(scores, descending=True, stable=True)
        return torch.cat([boxes[keep], scores[keep, None]], dim=-1), keep

    cfg = dict(nms_cfg)
    class_agnostic = cfg.pop('class_agnostic', class_agnostic)
    nms_type = cfg.pop('type', 'nms_rotated')
    if nms_type not in ('nms_rotated', 'rotated_nms', 'nms'):
        raise ValueError('batched_nms_rotated only supports type '
                         f"'nms_rotated' or 'nms', but got {nms_type!r}.")
    iou_threshold = cfg.pop('iou_threshold', cfg.pop('iou_thr', None))
    if iou_threshold is None:
        raise KeyError(
            'nms_cfg must contain iou_threshold (or the legacy iou_thr).')
    score_threshold = cfg.pop('score_threshold', cfg.pop('score_thr', 0.0))
    max_num = cfg.pop('max_num', -1)
    cfg.pop('split_thr', None)
    clockwise = cfg.pop('clockwise', True)
    angle_unit = cfg.pop('angle_unit', 'radian')
    if cfg:
        unknown = ', '.join(sorted(cfg))
        raise ValueError('Unsupported rotated NMS config keys: '
                         f'{unknown}. Input keys were: '
                         f'{", ".join(sorted(nms_cfg))}.')

    if score_threshold > 0:
        valid = scores > score_threshold
        original_index = torch.nonzero(valid, as_tuple=False).squeeze(1)
        boxes = boxes[valid]
        scores = scores[valid]
        idxs = idxs[valid] if idxs is not None else None
    else:
        original_index = None
    if boxes.size(0) == 0:
        return boxes.new_zeros((0, 6)), boxes.new_empty((0, ),
                                                        dtype=torch.long)

    labels = None if class_agnostic else idxs
    dets, keep = nms_rotated(
        boxes,
        scores,
        iou_threshold,
        labels=labels,
        clockwise=clockwise,
        angle_unit=angle_unit,
        max_num=max_num)
    if original_index is not None:
        keep = original_index[keep]
    return dets, keep


def multiclass_nms_rotated(multi_bboxes: Tensor,
                           multi_scores: Tensor,
                           score_thr: float,
                           nms_cfg: dict,
                           max_num: int = -1,
                           score_factors: Optional[Tensor] = None,
                           return_inds: bool = False) -> tuple:
    """Multi-class rotated NMS.

    Args:
        multi_bboxes (Tensor): Shape (N, C*5) or (N, 5).
        multi_scores (Tensor): Shape (N, C+1). The background score is last.
        score_thr (float): Score threshold applied before NMS.
        nms_cfg (dict): Rotated NMS configuration.
        max_num (int): Maximum detections after NMS.
        score_factors (Tensor, optional): Per-box multipliers for scores.
        return_inds (bool): Whether to return original box indices.

    Returns:
        tuple: ``(detections, labels)`` or
        ``(detections, labels, indices)``.
    """
    if multi_bboxes.dim() != 2 or (multi_bboxes.size(-1) != 5
                                   and multi_bboxes.size(-1) % 5 != 0):
        raise ValueError('multi_bboxes must have shape (N, C*5), but got '
                         f'{tuple(multi_bboxes.shape)}.')
    if multi_scores.dim() != 2:
        raise ValueError('multi_scores must have shape (N, C+1), but got '
                         f'{tuple(multi_scores.shape)}.')
    if multi_scores.size(0) != multi_bboxes.size(0):
        raise ValueError(
            'multi_bboxes and multi_scores must have the same first '
            f'dimension, but got {multi_bboxes.size(0)} and '
            f'{multi_scores.size(0)}.')

    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.size(1) > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]
    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    valid = scores > score_thr
    inds = valid.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if score_factors is not None:
        factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes).reshape(-1)
        scores = scores * factors[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], dim=-1)
        if return_inds:
            return dets, labels, inds
        return dets, labels

    dets, keep = batched_nms_rotated(bboxes, scores, labels, nms_cfg)
    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]
    if return_inds:
        return dets, labels[keep], inds[keep]
    return dets, labels[keep]


# Short aliases used by some downstream projects.
rotated_nms = nms_rotated
rotated_batched_nms = batched_nms_rotated

__all__ = [
    'nms_rotated', 'batched_nms_rotated', 'multiclass_nms_rotated',
    'rotated_nms', 'rotated_batched_nms'
]
