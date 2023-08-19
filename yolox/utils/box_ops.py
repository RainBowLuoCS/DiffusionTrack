# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
from yolox.utils.cluster_nms import giou_3d

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1,boxes2,boxes3,boxes4):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # boxes1=boxes1.float()
    # boxes2=boxes2.float()
    # boxes3=boxes3.float()
    # boxes4=boxes4.float()
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert (boxes3[:, 2:] >= boxes3[:, :2]).all()
    assert (boxes4[:, 2:] >= boxes4[:, :2]).all()
    # iou1, union1 = box_iou(boxes1, boxes3)
    # iou2, union2 = box_iou(boxes2, boxes4)
    # lt = torch.min(boxes1[:, None, :2], boxes3[:, :2])
    # rb = torch.max(boxes1[:, None, 2:], boxes3[:, 2:])

    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # area1 = wh[:, :, 0] * wh[:, :, 1]

    # lt = torch.min(boxes2[:, None, :2], boxes4[:, :2])
    # rb = torch.max(boxes2[:, None, 2:], boxes4[:, 2:])

    # wh = (rb - lt).clamp(min=0)  # [N,M,2]
    # area2 = wh[:, :, 0] * wh[:, :, 1]
    # uiou=(iou1*union1+iou2*union2)/(union1+union2)
    # uunion=union1+union2
    # uarea=area1+area2
    # return  uiou- (uarea - uunion) / uarea

    return giou_3d(boxes1,boxes3,boxes2,boxes4)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



# boxes = targets[:, :4].copy()
# labels = targets[:, 4].copy()
# ids = targets[:, 5].copy()
# if len(boxes) == 0:
#     targets = np.zeros((self.max_labels, 6), dtype=np.float32)
#     image, r_o = preproc(image, input_dim, self.means, self.std)
#     image = np.ascontiguousarray(image, dtype=np.float32)
#     return image, targets

# image_o = image.copy()
# targets_o = targets.copy()
# height_o, width_o, _ = image_o.shape
# boxes_o = targets_o[:, :4]
# labels_o = targets_o[:, 4]
# ids_o = targets_o[:, 5]
# # bbox_o: [xyxy] to [c_x,c_y,w,h]
# boxes_o = xyxy2cxcywh(boxes_o)

# image_t = _distort(image)
# image_t, boxes_t ,image_r,boxes_r= _mirror(image_t, boxes)
# height, width, _ = image_t.shape
# image_t, r_t = preproc(image_t, input_dim, self.means, self.std)
# image_t, r_r = preproc(image_r, input_dim, self.means, self.std)
# # boxes [xyxy] 2 [cx,cy,w,h]
# boxes_t = xyxy2cxcywh(boxes_t)
# boxes_t *= r_t

# boxes_r = xyxy2cxcywh(boxes_r)
# boxes_r *= r_r

# mask_b = np.minimum(boxes_t[:, 2], boxes_t[:, 3]) > 1
# boxes_t = boxes_t[mask_b]
# boxes_r = boxes_r[mask_b]

# labels_t = labels[mask_b]
# ids_t = ids[mask_b]

# if len(boxes_t) == 0:
#     image_t, r_o = preproc(image_o, input_dim, self.means, self.std)
#     boxes_o *= r_o
#     boxes_t = boxes_o
#     image_r=image_t
#     boxes_r=boxes_t
#     labels_t = labels_o
#     ids_t = ids_o

# labels_t = np.expand_dims(labels_t, 1)
# ids_t = np.expand_dims(ids_t, 1)

# targets_t = np.hstack((labels_t, boxes_t, ids_t))
# padded_labels = np.zeros((self.max_labels, 6))
# padded_labels[range(len(targets_t))[: self.max_labels]] = targets_t[
#     : self.max_labels
# ]

# targets_r = np.hstack((labels_t, boxes_r, ids_t))
# padded_labels_r = np.zeros((self.max_labels, 6))
# padded_labels_r[range(len(targets_r))[: self.max_labels]] = targets_r[
#     : self.max_labels
# ]
# padded_labels = np.ascontiguousarray(padded_labels, dtype=np.float32)
# image_t = np.ascontiguousarray(image_t, dtype=np.float32)
# return image_t, padded_labels
