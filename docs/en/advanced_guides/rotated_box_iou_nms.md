# Rotated Box IoU and NMS

MMDetection provides pure-PyTorch rotated-box IoU and NMS operators. They use
the same code path on CPU and CUDA, so the result does not depend on an
optional compiled extension.

## Box Format

Rotated boxes use five values per row:

```text
(cx, cy, w, h, angle)
```

- `cx`, `cy`: center coordinates.
- `w`, `h`: non-negative width and height.
- `angle`: radians by default. Set `angle_unit='degree'` to use degrees.

The positive-angle direction follows MMCV/OpenCV by default (`clockwise=True`).
Use `clockwise=False` for counter-clockwise angles. Periodically equivalent
angles, such as `angle + pi` and `(w, h, angle + pi/2)`, produce the same IoU.

## IoU API

```python
from mmdet.models.layers import box_iou_rotated

ious = box_iou_rotated(boxes1, boxes2)                 # (N, M)
aligned = box_iou_rotated(boxes1, boxes2, aligned=True)  # (N,)
iof = box_iou_rotated(boxes1, boxes2, mode='iof')
batched = box_iou_rotated(boxes1_batched, boxes2_batched)
```

Important parameters:

- `aligned`: compute only corresponding pairs; both inputs must contain the
  same number of boxes.
- `mode`: `'iou'` or `'iof'`.
- `clockwise`: angle direction convention.
- `angle_unit`: `'radian'` or `'degree'`.
- `eps`: positive denominator clamp used for numerical stability.

`normalize_angle` wraps an angle into the `oc`, `le90`, `le135`, or `r360`
range. `normalize_rotated_boxes` additionally swaps width and height when
needed to produce a canonical box representation.

## NMS APIs

```python
from mmdet.models.layers import (batched_nms_rotated,
                                 multiclass_nms_rotated, nms_rotated)

dets, keep = nms_rotated(boxes, scores, 0.5)
dets, keep = batched_nms_rotated(
    boxes, scores, labels, dict(type='nms_rotated', iou_threshold=0.5))
dets, labels = multiclass_nms_rotated(
    boxes, class_scores, 0.05,
    dict(type='nms_rotated', iou_threshold=0.5))
```

`nms_rotated` returns boxes with scores appended, shape `(K, 6)`, and the
original indices. Suppression uses `IoU >= iou_threshold`, matching MMCV.
Passing `labels` suppresses boxes only within the same label.

The horizontal `multiclass_nms` interface remains unchanged. Passing
`box_dim=5` automatically dispatches to `batched_nms_rotated`.

## Detection Head

`RotatedRetinaHead` uses:

- `RotatedAnchorGenerator` with zero-angle base anchors.
- `DeltaXYWHTRBBoxCoder` with five regression deltas.
- Automatic rotated NMS in dense-head post-processing when boxes have five
  values per row.

Training assignment can use:

```python
iou_calculator=dict(type='RotatedBboxOverlaps2D')
```

`RotatedBoxes` is registered as box type `'rbox'` and supports conversion to
and from horizontal boxes. Its `overlaps` method calls the operator described
above.

## Error Handling

Invalid shapes, non-finite values, negative width/height, mixed devices,
mixed dtypes, non-broadcastable batch dimensions, and invalid thresholds raise
an exception that includes the argument name, input shape/device, and the
first offending index where applicable. Empty boxes represented as `(0, 5)`
are supported and return empty outputs.

## Tests

Run the focused tests with:

```bash
pytest -q tests/test_models/test_layers/test_rotated_box_iou_nms.py \
       tests/test_models/test_dense_heads/test_rotated_retina_head.py
```

The tests cover reference fixtures, angle periods, boundary intersections,
empty inputs, batched inputs, multiclass NMS, and CPU/CUDA consistency. The
10k-box sparse benchmark asserts a 50 ms upper bound after warm-up and is
enabled explicitly to avoid flaky timing on shared CI machines:

```bash
MMDET_RUN_ROTATED_NMS_BENCHMARK=1 \
  pytest -q tests/test_models/test_layers/test_rotated_box_iou_nms.py
```

## Known Limitations

- The implementation is intended for convex rectangles. General polygons and
  rotated quadrilaterals are not supported.
- Greedy NMS requires pairwise comparisons in the worst case when all boxes
  overlap. The common sparse case uses a sweep-line broad phase and is much
  faster, but pathological dense inputs remain quadratic.
- ONNX/TensorRT export of the rotated operators is not provided.
- CPU and CUDA use identical operations, but floating-point reduction order
  can still introduce tiny differences. Tests compare with a small tolerance
  and require identical kept indices for the supplied fixtures.
