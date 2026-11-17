# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from .retina_head import RetinaHead


@MODELS.register_module()
class RotatedRetinaHead(RetinaHead):
    """Retina-style head for rotated boxes.

    The regression branch predicts ``(dx, dy, dw, dh, dangle)`` and the
    post-processing path automatically uses rotated NMS when the box tensor
    has five values per row.

    Args:
        anchor_generator (dict): Rotated anchor generator config. Defaults
            to :class:`RotatedAnchorGenerator` with the usual Retina anchors.
        bbox_coder (dict): Rotated delta coder config. Defaults to
            :class:`DeltaXYWHTRBBoxCoder`.
        **kwargs: Arguments forwarded to :class:`RetinaHead`.
    """

    def __init__(self,
                 anchor_generator=dict(
                     type='RotatedAnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128],
                     angle_version='le90'),
                 bbox_coder=dict(
                     type='DeltaXYWHTRBBoxCoder',
                     angle_version='le90',
                     use_box_type=True),
                 **kwargs):
        super().__init__(
            anchor_generator=anchor_generator, bbox_coder=bbox_coder, **kwargs)


__all__ = ['RotatedRetinaHead']
