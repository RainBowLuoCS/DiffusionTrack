import torch

@torch.jit.script
def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter

@torch.jit.script
def garea(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.max(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
    min_xy = torch.min(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                       box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
    return torch.clamp(max_xy - min_xy, min=0).prod(3)  # inter

@torch.jit.script
def get_box_area(box):
    return (box[:, :, 2]-box[:, :, 0]) *(box[:, :, 3]-box[:, :, 1])

def giou_3d(box_a,box_b,box_c,box_d):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]
        box_c = box_c[None, ...]
        box_d = box_d[None, ...]

    interab = intersect(box_a,box_b)
    intercd = intersect(box_c,box_d)
    
    area_ab= garea(box_a,box_b)
    area_cd=garea(box_c,box_d)

    area_a = get_box_area(box_a).unsqueeze(2).expand_as(interab)  # [A,B]
    area_b = get_box_area(box_b).unsqueeze(1).expand_as(interab)  # [A,B]
    area_c = get_box_area(box_c).unsqueeze(2).expand_as(intercd)  # [A,B]
    area_d = get_box_area(box_d).unsqueeze(1).expand_as(intercd)  # [A,B]
    unionab = area_a + area_b - interab
    unioncd = area_c+area_d-intercd

    uiouabcd = (interab+intercd) / (unionab+unioncd)
    out=uiouabcd-(area_ab+area_cd-unionab-unioncd)/(area_ab+area_cd)
    return out if use_batch else out.squeeze(0)

def cluster_nms(boxes_a,boxes_c,scores,iou_threshold:float=0.5, top_k:int=500):
    # Collapse all the classes into 1 
    _, idx = scores.sort(0, descending=True)
    idx = idx[:top_k]
    boxes_a = boxes_a[idx]
    boxes_b = boxes_a
    boxes_c = boxes_c[idx]
    boxes_d = boxes_c
    iou = giou_3d(boxes_a,boxes_b,boxes_c,boxes_d).triu_(diagonal=1)
    B = iou
    for i in range(200):
        A=B
        maxA,_=torch.max(A, dim=0)
        E = (maxA<=iou_threshold).float().unsqueeze(1).expand_as(A)
        B=iou.mul(E)
        if A.equal(B)==True:
            break
    idx_out = idx[maxA <= iou_threshold]
    return idx_out



# ## test

# boxes_a=[[100,100,200,200],
#          [110,110,210,210],
#          [50,50,150,150],
#          [100,100,200,200],
#          [90,90,190,190],]

# boxes_c=[[100,100,200,200],
#          [110,110,210,210],
#          [150,150,250,250],
#          [0,0,100,100],
#          [10,10,110,110],]

# scores=[0.91,0.9,0.95,0.9,0.8]

# boxes_a=torch.tensor(boxes_a,dtype=torch.float)
# boxes_c=torch.tensor(boxes_c,dtype=torch.float)
# scores=torch.tensor(scores,dtype=torch.float)


# indix=cluster_nms(boxes_a,boxes_c,scores)
# print(indix)
