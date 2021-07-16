import torch

import mmdeploy
from mmdeploy.core import FUNCTION_REWRITER
from mmdeploy.mmcv.ops import DummyONNXNMSop, TRTBatchedNMSop


def select_nms_index(scores, boxes, nms_index, batch_size, keep_top_k=-1):
    batch_inds, cls_inds = nms_index[:, 0], nms_index[:, 1]
    box_inds = nms_index[:, 2]

    # index by nms output
    scores = scores[batch_inds, cls_inds, box_inds].unsqueeze(1)
    boxes = boxes[batch_inds, box_inds, ...]
    dets = torch.cat([boxes, scores], dim=1)

    # batch all
    batched_dets = dets.unsqueeze(0).repeat(batch_size, 1, 1)
    batch_template = torch.arange(
        0, batch_size, dtype=batch_inds.dtype, device=batch_inds.device)
    batched_dets = batched_dets.where(
        (batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),
        batched_dets.new_zeros(1))

    batched_labels = cls_inds.unsqueeze(0).repeat(batch_size, 1)
    batched_labels = batched_labels.where(
        (batch_inds == batch_template.unsqueeze(1)),
        batched_labels.new_ones(1) * -1)

    # sort
    if keep_top_k > 0:
        _, topk_inds = batched_dets[:, :, -1].topk(keep_top_k, dim=1)
    else:
        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)
    topk_batch_inds = torch.arange(
        batch_size, dtype=topk_inds.dtype,
        device=topk_inds.device).view(-1, 1).expand_as(topk_inds)
    batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
    batched_labels = batched_labels[topk_batch_inds, topk_inds, ...]

    return batched_dets, batched_labels


def _multiclass_nms(boxes,
                    scores,
                    max_output_boxes_per_class=1000,
                    iou_threshold=0.5,
                    score_threshold=0.05,
                    pre_top_k=-1,
                    keep_top_k=-1,
                    labels=None):
    """Create a dummy onnx::NonMaxSuppression op while exporting to ONNX.

    This function helps exporting to onnx with batch and multiclass NMS op.
    It only supports class-agnostic detection results. That is, the scores
    is of shape (N, num_bboxes, num_classes) and the boxes is of shape
    (N, num_boxes, 4).

    Args:
        boxes (Tensor): The bounding boxes of shape [N, num_boxes, 4]
        scores (Tensor): The detection scores of shape
            [N, num_boxes, num_classes]
        max_output_boxes_per_class (int): Maximum number of output
            boxes per class of nms. Defaults to 1000.
        iou_threshold (float): IOU threshold of nms. Defaults to 0.5
        score_threshold (float): score threshold of nms.
            Defaults to 0.05.
        pre_top_k (bool): Number of top K boxes to keep before nms.
            Defaults to -1.
        keep_top_k (int): Number of top K boxes to keep after nms.
            Defaults to -1.
        labels (Tensor, optional): It not None, explicit labels would be used.
            Otherwise, labels would be automatically generated using
            num_classed. Defaults to None.

    Returns:
        tuple[Tensor, Tensor]: dets of shape [N, num_det, 5] and class labels
            of shape [N, num_det].
    """
    max_output_boxes_per_class = torch.LongTensor([max_output_boxes_per_class])
    iou_threshold = torch.tensor([iou_threshold], dtype=torch.float32)
    score_threshold = torch.tensor([score_threshold], dtype=torch.float32)
    batch_size = scores.shape[0]
    num_class = scores.shape[2]

    if pre_top_k > 0:
        max_scores, _ = scores.max(-1)
        _, topk_inds = max_scores.topk(pre_top_k)
        batch_inds = torch.arange(batch_size).view(
            -1, 1).expand_as(topk_inds).long()
        # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
        boxes = boxes[batch_inds, topk_inds, :]
        scores = scores[batch_inds, topk_inds, :]
        if labels is not None:
            labels = labels[batch_inds, topk_inds]

    scores = scores.permute(0, 2, 1)
    selected_indices = DummyONNXNMSop.apply(boxes, scores,
                                            max_output_boxes_per_class,
                                            iou_threshold, score_threshold)

    if labels is None:
        labels = torch.arange(num_class, dtype=torch.long).to(scores.device)
        labels = labels.view(1, num_class, 1).expand_as(scores)

    dets, labels = select_nms_index(
        scores, boxes, selected_indices, batch_size, keep_top_k=keep_top_k)
    return dets, labels


@FUNCTION_REWRITER.register_rewriter(
    func_name='mmdeploy.mmdet.core.post_processing._multiclass_nms',
    backend='tensorrt')
def multiclass_nms_static(ctx,
                          boxes,
                          scores,
                          max_output_boxes_per_class=1000,
                          iou_threshold=0.5,
                          score_threshold=0.05,
                          pre_top_k=-1,
                          keep_top_k=-1,
                          labels=None):
    boxes = boxes if boxes.dim() == 4 else boxes.unsqueeze(2)
    keep_top_k = max_output_boxes_per_class if keep_top_k < 0 else min(
        max_output_boxes_per_class, keep_top_k)
    dets, labels = TRTBatchedNMSop.apply(boxes, scores, int(scores.shape[-1]),
                                         pre_top_k, keep_top_k, iou_threshold,
                                         score_threshold, -1)

    return dets, labels


def multiclass_nms(*args, **kwargs):
    """Wrapper function for _multiclass_nms."""
    return mmdeploy.mmdet.core.post_processing._multiclass_nms(*args, **kwargs)