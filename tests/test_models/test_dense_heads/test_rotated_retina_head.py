# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData

from mmdet.models.dense_heads import RotatedRetinaHead
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import RotatedBoxes


def test_rotated_retina_head_forward_and_postprocess():
    head = RotatedRetinaHead(
        num_classes=2, in_channels=8, feat_channels=8, stacked_convs=1)
    head.eval()
    with torch.no_grad():
        cls_scores, bbox_preds = head((torch.randn(1, 8, 16, 16), ))
    assert cls_scores[0].shape[0] == 1
    assert bbox_preds[0].shape[1] % head.bbox_coder.encode_size == 0
    assert head.bbox_coder.encode_size == 5

    results = InstanceData()
    results.bboxes = RotatedBoxes(
        torch.tensor([[0.0, 0.0, 4.0, 4.0, 0.0], [0.1, 0.1, 4.0, 4.0, 0.0]]))
    results.scores = torch.tensor([0.9, 0.8])
    results.labels = torch.tensor([0, 0])
    cfg = ConfigDict(
        nms=dict(type='nms_rotated', iou_threshold=0.5),
        max_per_img=10,
        min_bbox_size=-1)
    output = head._bbox_post_process(
        results=results,
        cfg=cfg,
        rescale=False,
        img_meta=dict(scale_factor=[1.0, 1.0, 1.0, 1.0]))
    assert len(output) == 1
    assert output.bboxes.size(-1) == 5


def test_rotated_retina_head_loss_and_predict():
    head = RotatedRetinaHead(
        num_classes=2,
        in_channels=8,
        feat_channels=8,
        stacked_convs=1,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            strides=[8],
            ratios=[1.0],
            scales=[1.0]),
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0.0,
                match_low_quality=True,
                iou_calculator=dict(type='RotatedBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=ConfigDict(
            nms_pre=1000,
            min_bbox_size=-1,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.5),
            max_per_img=100))
    data_sample = DetDataSample()
    data_sample.set_metainfo(
        dict(
            img_shape=(64, 64),
            pad_shape=(64, 64),
            scale_factor=(1.0, 1.0, 1.0, 1.0)))
    gt_instances = InstanceData()
    gt_instances.bboxes = RotatedBoxes(
        torch.tensor([[20.0, 20.0, 10.0, 5.0, 0.2]]))
    gt_instances.labels = torch.tensor([0])
    data_sample.gt_instances = gt_instances
    features = (torch.randn(1, 8, 8, 8), )

    losses = head.loss(features, [data_sample])
    assert set(losses) == {'loss_cls', 'loss_bbox'}
    assert all(torch.isfinite(loss[0]) for loss in losses.values())

    head.eval()
    with torch.no_grad():
        predictions = head.predict(features, [data_sample])
    assert len(predictions) == 1
    assert predictions[0].bboxes.size(-1) == 5
    assert predictions[0].scores.shape == predictions[0].labels.shape
