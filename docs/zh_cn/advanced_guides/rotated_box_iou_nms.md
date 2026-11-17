# 旋转框 IoU 与 NMS

MMDetection 提供纯 PyTorch 的旋转框 IoU 与 NMS 算子。CPU 与 CUDA 使用同一计算路径，
不依赖可选的编译扩展。

## 框格式

每个旋转框包含五个值：

```text
(cx, cy, w, h, angle)
```

- `cx`、`cy`：中心点坐标。
- `w`、`h`：非负宽度和高度。
- `angle`：默认使用弧度；可设置 `angle_unit='degree'` 使用角度。

默认角度正方向与 MMCV/OpenCV 一致（`clockwise=True`）。使用逆时针定义时传入
`clockwise=False`。周期等价角度（如 `angle + pi`，或交换宽高并加 `pi/2`）计算出的
IoU 相同。

## IoU 接口

```python
from mmdet.models.layers import box_iou_rotated

ious = box_iou_rotated(boxes1, boxes2)                    # (N, M)
aligned = box_iou_rotated(boxes1, boxes2, aligned=True)  # (N,)
iof = box_iou_rotated(boxes1, boxes2, mode='iof')
batched = box_iou_rotated(boxes1_batched, boxes2_batched)
```

主要参数：

- `aligned`：只计算对应位置框的 IoU，要求输入框数量相同。
- `mode`：`'iou'` 或 `'iof'`。
- `clockwise`：角度正方向。
- `angle_unit`：`'radian'` 或 `'degree'`。
- `eps`：分母下界，用于数值稳定。

`normalize_angle` 将角度归一化到 `oc`、`le90`、`le135` 或 `r360` 区间；
`normalize_rotated_boxes` 还会在需要时交换宽高，得到规范表示。

## NMS 接口

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

`nms_rotated` 返回追加分数后的框，形状为 `(K, 6)`，并返回原始索引。抑制条件为
`IoU >= iou_threshold`，与 MMCV 保持一致。传入 `labels` 时只在同标签框之间抑制。

既有水平框 `multiclass_nms` 接口保持不变；传入 `box_dim=5` 时自动分派到
`batched_nms_rotated`。

## 检测头

`RotatedRetinaHead` 使用：

- `RotatedAnchorGenerator` 生成零角度基础锚框。
- `DeltaXYWHTRBBoxCoder` 回归五个偏移量。
- 稠密检测头后处理检测到每行 5 个值时自动调用旋转框 NMS。

训练分配器可使用：

```python
iou_calculator=dict(type='RotatedBboxOverlaps2D')
```

`RotatedBoxes` 以 `'rbox'` 名称注册，支持与水平框相互转换；其 `overlaps` 方法调用
上述旋转框 IoU 算子。

## 错误处理

形状错误、非有限值、负宽高、设备或数据类型不一致、批量维度不可广播以及非法阈值
都会抛出异常，信息包含参数名、输入形状/设备，并在适用时包含首个错误索引。空框
`(0, 5)` 可正常计算并返回空结果，不会崩溃。

## 测试

```bash
pytest -q tests/test_models/test_layers/test_rotated_box_iou_nms.py \
       tests/test_models/test_dense_heads/test_rotated_retina_head.py
```

测试覆盖参考数值、角度周期、边界相交、空输入、批量输入、多类别 NMS 和 CPU/CUDA
一致性。预热后的一万框稀疏 NMS 基准要求不超过 50 ms；为避免共享 CI 机器上的计时
抖动，该基准需显式开启：

```bash
MMDET_RUN_ROTATED_NMS_BENCHMARK=1 \
  pytest -q tests/test_models/test_layers/test_rotated_box_iou_nms.py
```

## 已知限制

- 仅支持凸矩形，不支持任意多边形。
- 最坏情况下所有框都重叠时，贪心 NMS 仍需二次比较；常见稀疏输入通过扫描线粗筛，
  复杂度接近线性。
- 暂不提供旋转算子的 ONNX/TensorRT 导出。
- CPU 与 CUDA 使用相同运算，但浮点规约顺序仍可能产生极小差异。测试使用小容差，
  并要求夹具上的保留索引完全一致。
