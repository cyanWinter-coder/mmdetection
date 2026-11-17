# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import time

import pytest
import torch
from torch.testing import assert_close

from mmdet.models.layers import (batched_nms_rotated, box_iou_rotated,
                                 multiclass_nms_rotated, nms_rotated)


def _fixture_boxes():
    boxes1 = torch.tensor(
        [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
         [7.0, 7.0, 8.0, 8.0, 0.4]],
        dtype=torch.float32)
    boxes2 = torch.tensor(
        [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
         [5.0, 5.0, 6.0, 7.0, 0.4]],
        dtype=torch.float32)
    return boxes1, boxes2


def test_box_iou_rotated_reference():
    boxes1, boxes2 = _fixture_boxes()
    expected_iou = torch.tensor(
        [[0.3708, 0.4351, 0.0000], [0.1104, 0.4487, 0.0424],
         [0.0000, 0.0000, 0.3622]],
        dtype=torch.float32)
    expected_iof = torch.tensor(
        [[0.4959, 0.5306, 0.0000], [0.1823, 0.5420, 0.1832],
         [0.0000, 0.0000, 0.4404]],
        dtype=torch.float32)

    assert_close(
        box_iou_rotated(boxes1, boxes2), expected_iou, atol=1e-4, rtol=0)
    assert_close(
        box_iou_rotated(boxes1, boxes2, aligned=True),
        torch.tensor([0.3708, 0.4487, 0.3622]),
        atol=1e-4,
        rtol=0)
    assert_close(
        box_iou_rotated(boxes1, boxes2, mode='iof'),
        expected_iof,
        atol=1e-4,
        rtol=0)


def test_box_iou_rotated_angle_period_and_units():
    boxes1 = torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.3]])
    boxes2 = torch.tensor([[1.0, 0.2, 3.0, 2.5, -0.4]])
    expected = box_iou_rotated(boxes1, boxes2)

    equivalent = [
        torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.3 + math.pi]]),
        torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.3 - math.pi]]),
        torch.tensor([[0.0, 0.0, 4.0, 2.0, 0.3 + 100 * math.pi]]),
        torch.tensor([[0.0, 0.0, 2.0, 4.0, 0.3 + math.pi / 2]]),
    ]
    for boxes in equivalent:
        assert_close(
            box_iou_rotated(boxes, boxes2), expected, atol=1e-5, rtol=0)

    boxes1_degree = boxes1.clone()
    boxes2_degree = boxes2.clone()
    boxes1_degree[:, 4] *= 180 / math.pi
    boxes2_degree[:, 4] *= 180 / math.pi
    assert_close(
        box_iou_rotated(boxes1_degree, boxes2_degree, angle_unit='degree'),
        expected,
        atol=1e-5,
        rtol=0)


def test_box_iou_rotated_boundary_and_empty():
    touching = box_iou_rotated(
        torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]]),
        torch.tensor([[2.0, 0.0, 2.0, 2.0, 0.0]]))
    assert_close(touching, torch.zeros(1, 1))

    crossing = box_iou_rotated(
        torch.tensor([[0.0, 0.0, 4.0, 2.0, math.pi / 4]]),
        torch.tensor([[0.0, 0.0, 4.0, 2.0, -math.pi / 4]]))
    assert crossing.item() > 0

    empty = torch.empty(0, 5)
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
    assert box_iou_rotated(empty, boxes).shape == (0, 1)
    assert box_iou_rotated(boxes, empty).shape == (1, 0)
    assert box_iou_rotated(empty, empty).shape == (0, 0)
    assert box_iou_rotated(empty, empty, aligned=True).shape == (0, )


def test_box_iou_rotated_batch():
    boxes1, boxes2 = _fixture_boxes()
    batch1 = torch.stack([boxes1, boxes1.flip(0)])
    batch2 = torch.stack([boxes2, boxes2.flip(0)])
    batched = box_iou_rotated(batch1, batch2)
    assert batched.shape == (2, 3, 3)
    assert_close(batched[0], box_iou_rotated(batch1[0], batch2[0]))
    assert_close(batched[1], box_iou_rotated(batch1[1], batch2[1]))


def test_box_iou_rotated_random_symmetry():
    generator = torch.Generator().manual_seed(7)
    boxes1 = torch.rand(32, 5, generator=generator)
    boxes2 = torch.rand(24, 5, generator=generator)
    boxes1[:, 2:4] = boxes1[:, 2:4] * 20 + 1
    boxes2[:, 2:4] = boxes2[:, 2:4] * 20 + 1
    boxes1[:, 4] = (boxes1[:, 4] - 0.5) * 2 * math.pi
    boxes2[:, 4] = (boxes2[:, 4] - 0.5) * 2 * math.pi
    overlaps = box_iou_rotated(boxes1, boxes2)
    assert_close(overlaps, box_iou_rotated(boxes2, boxes1).t())
    assert bool(((overlaps >= 0) & (overlaps <= 1)).all())
    self_overlaps = box_iou_rotated(boxes1, boxes1, aligned=True)
    assert_close(self_overlaps, torch.ones_like(self_overlaps))


def test_box_iou_rotated_errors_include_context():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
    invalid = boxes.clone()
    invalid[0, 2] = -1
    with pytest.raises(ValueError, match='boxes1.*negative'):
        box_iou_rotated(invalid, boxes)

    non_finite = boxes.clone()
    non_finite[0, 0] = float('nan')
    with pytest.raises(ValueError, match='boxes1.*non-finite'):
        box_iou_rotated(non_finite, boxes)

    with pytest.raises(ValueError, match='broadcastable'):
        box_iou_rotated(boxes.expand(2, 1, 5), boxes.expand(3, 1, 5))


def test_nms_rotated_reference():
    boxes = torch.tensor([[6.0, 3.0, 8.0, 7.0, 0.5],
                          [3.0, 6.0, 9.0, 11.0, 0.6],
                          [3.0, 7.0, 10.0, 12.0, 0.3],
                          [1.0, 4.0, 13.0, 7.0, 0.6]])
    scores = torch.tensor([0.7, 0.8, 0.5, 0.9])

    dets, keep = nms_rotated(boxes, scores, 0.5)
    assert keep.tolist() == [3, 1, 0]
    assert_close(dets[:, :5], boxes[keep])
    assert_close(dets[:, 5], scores[keep])

    ccw_boxes = boxes.clone()
    ccw_boxes[:, 4] *= -1
    _, keep = nms_rotated(ccw_boxes, scores, 0.5, clockwise=False)
    assert keep.tolist() == [3, 1, 0]

    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    _, keep = nms_rotated(boxes, scores, 0.5, labels=labels)
    assert keep.tolist() == [3, 1, 0, 2]

    _, keep = batched_nms_rotated(boxes, scores, labels,
                                  dict(type='nms_rotated', iou_threshold=0.5))
    assert keep.tolist() == [3, 1, 0, 2]

    _, keep = batched_nms_rotated(
        boxes,
        scores,
        labels,
        dict(type='nms_rotated', iou_threshold=0.5),
        class_agnostic=True)
    assert keep.tolist() == [3, 1, 0]


def test_nms_rotated_boundary_and_empty():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0], [2.0, 0.0, 2.0, 2.0,
                                                      0.0]])
    scores = torch.tensor([0.9, 0.8])
    _, keep = nms_rotated(boxes, scores, 0.5)
    assert keep.tolist() == [0, 1]
    _, keep = nms_rotated(boxes, scores, 0.0)
    assert keep.tolist() == [0]

    empty_boxes = torch.empty(0, 5)
    empty_scores = torch.empty(0)
    dets, keep = nms_rotated(empty_boxes, empty_scores, 0.5)
    assert dets.shape == (0, 6)
    assert keep.shape == (0, )


def test_multiclass_nms_rotated():
    boxes = torch.tensor([
        [0.0, 0.0, 4.0, 4.0, 0.0],
        [0.2, 0.2, 4.0, 4.0, 0.0],
        [10.0, 10.0, 4.0, 4.0, 0.0],
    ])
    scores = torch.tensor([[0.9, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.95, 0.0]])
    dets, labels = multiclass_nms_rotated(
        boxes, scores, 0.05, dict(type='nms_rotated', iou_threshold=0.5))
    assert dets.shape == (2, 6)
    assert labels.tolist() == [1, 0]


def test_nms_rotated_large_sparse_input():
    """10k well-separated boxes should be handled correctly."""
    num_boxes = 10000
    grid_size = 100
    y_grid, x_grid = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    boxes = torch.stack([
        x_grid.flatten() * 100.0 + 50.0,
        y_grid.flatten() * 100.0 + 50.0,
        torch.full((num_boxes, ), 10.0),
        torch.full((num_boxes, ), 20.0),
        torch.zeros(num_boxes),
    ],
                        dim=-1)
    scores = torch.linspace(0.1, 1.0, num_boxes)

    _, keep = nms_rotated(boxes, scores, 0.5)
    assert len(keep) == num_boxes


@pytest.mark.skipif(
    os.getenv('MMDET_RUN_ROTATED_NMS_BENCHMARK') != '1',
    reason='Set MMDET_RUN_ROTATED_NMS_BENCHMARK=1 to run timing checks.')
def test_nms_rotated_10k_benchmark():
    num_boxes = 10000
    grid_size = 100
    y_grid, x_grid = torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing='ij')
    boxes = torch.stack([
        x_grid.flatten() * 100.0 + 50.0,
        y_grid.flatten() * 100.0 + 50.0,
        torch.full((num_boxes, ), 10.0),
        torch.full((num_boxes, ), 20.0),
        torch.zeros(num_boxes),
    ],
                        dim=-1)
    scores = torch.linspace(0.1, 1.0, num_boxes)

    # Warm up the same large-input path before timing.
    _, keep = nms_rotated(boxes, scores, 0.5)
    assert len(keep) == num_boxes
    elapsed_ms = []
    for _ in range(3):
        start = time.perf_counter()
        _, keep = nms_rotated(boxes, scores, 0.5)
        elapsed_ms.append((time.perf_counter() - start) * 1000)
        assert len(keep) == num_boxes
    elapsed_ms = min(elapsed_ms)
    assert elapsed_ms < 50, f'10k rotated NMS took {elapsed_ms:.2f} ms'


def test_nms_rotated_device_consistency():
    if not torch.cuda.is_available():
        return
    boxes = torch.tensor([[6.0, 3.0, 8.0, 7.0, 0.5],
                          [3.0, 6.0, 9.0, 11.0, 0.6],
                          [3.0, 7.0, 10.0, 12.0, 0.3],
                          [1.0, 4.0, 13.0, 7.0, 0.6]])
    scores = torch.tensor([0.7, 0.8, 0.5, 0.9])
    cpu_ious = box_iou_rotated(boxes, boxes)
    cuda_ious = box_iou_rotated(boxes.cuda(), boxes.cuda()).cpu()
    assert_close(cuda_ious, cpu_ious, atol=1e-5, rtol=1e-5)

    _, cpu_keep = nms_rotated(boxes, scores, 0.5)
    _, cuda_keep = nms_rotated(boxes.cuda(), scores.cuda(), 0.5)
    assert cpu_keep.tolist() == cuda_keep.cpu().tolist()
