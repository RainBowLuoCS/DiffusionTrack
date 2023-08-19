import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit
import torchvision.ops as ops
from yolox.utils import box_ops
from yolox.utils.dist import get_world_size, is_dist_avail_and_initialized
from yolox.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou


class SetCriterionDynamicK(nn.Module):
    """ This class computes the loss for DiffusionDet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self,num_classes, matcher, weight_dict, eos_coef, losses, use_focal,use_fed_loss):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        if self.use_fed_loss:
            self.fed_loss_num_classes = 50
            from detectron2.data.detection_utils import get_fed_loss_cls_weights
            cls_weight_fun = lambda: get_fed_loss_cls_weights(dataset_names=cfg.DATASETS.TRAIN, freq_weight_power=cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT_POWER)  # noqa
            fed_loss_cls_weights = cls_weight_fun()
            assert (
                    len(fed_loss_cls_weights) == self.num_classes
            ), "Please check the provided fed_loss_cls_weights. Their size should match num_classes"
            self.register_buffer("fed_loss_cls_weights", fed_loss_cls_weights)

        if self.use_focal:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0
        else:
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = self.eos_coef
            self.register_buffer('empty_weight', empty_weight)

    # copy-paste from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/fast_rcnn.py#L356
    def get_fed_loss_classes(self, gt_classes, num_fed_loss_classes, num_classes, weight):
        """
        Args:
            gt_classes: a long tensor of shape R that contains the gt class label of each proposal.
            num_fed_loss_classes: minimum number of classes to keep when calculating federated loss.
            Will sample negative classes if number of unique gt_classes is smaller than this value.
            num_classes: number of foreground classes
            weight: probabilities used to sample negative classes
        Returns:
            Tensor:
                classes to keep when calculating the federated loss, including both unique gt
                classes and sampled negative classes.
        """
        unique_gt_classes = torch.unique(gt_classes)
        prob = unique_gt_classes.new_ones(num_classes + 1).float()
        prob[-1] = 0
        if len(unique_gt_classes) < num_fed_loss_classes:
            prob[:num_classes] = weight.float().clone()
            prob[unique_gt_classes] = 0
            sampled_negative_classes = torch.multinomial(
                prob, num_fed_loss_classes - len(unique_gt_classes), replacement=False
            )
            fed_loss_classes = torch.cat([unique_gt_classes, sampled_negative_classes])
        else:
            fed_loss_classes = unique_gt_classes
        return fed_loss_classes

    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        conf_score=torch.cat([outputs['pred_scores'],outputs['pred_scores']],dim=0)
        p=torch.sqrt(torch.sigmoid(src_logits)*conf_score)
        src_logits=torch.log(p/(1-p))
        batch_size = len(targets)

        # src_logits_re=torch.cat((src_logits[:batch_size//2],src_logits[batch_size//2:]),dim=0)
        # src_logits=(src_logits+src_logits_re)/2

        # idx = self._get_src_permutation_idx(indices)
        # target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # src_logits_list = []
        target_classes_o_list = []
        # target_classes[idx] = target_classes_o
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx%(batch_size//2)][0]
            gt_multi_idx = indices[batch_idx%(batch_size//2)][1]
            if len(gt_multi_idx) == 0:
                continue
            # bz_src_logits = src_logits[batch_idx]
            target_classes_o = targets[batch_idx]["labels"]
            target_classes[batch_idx, valid_query] = target_classes_o[gt_multi_idx]

            # src_logits_list.append(bz_src_logits[valid_query])
            target_classes_o_list.append(target_classes_o[gt_multi_idx])

        if self.use_focal or self.use_fed_loss:
            num_boxes = torch.cat(target_classes_o_list).shape[0] if len(target_classes_o_list) != 0 else 1

            target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], self.num_classes + 1],
                                                dtype=src_logits.dtype, layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            loss_ce=0
            gt_classes = torch.argmax(target_classes_onehot, dim=-1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]
            target_classes_onehot = target_classes_onehot.flatten(0, 1)
            src_logits = src_logits.flatten(0, 1)
            if self.use_focal:
                cls_loss = sigmoid_focal_loss_jit(src_logits, target_classes_onehot, alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="none")
            else:
                cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes_onehot, reduction="none")
            if self.use_fed_loss:
                K = self.num_classes
                N = src_logits.shape[0]
                fed_loss_classes = self.get_fed_loss_classes(
                    gt_classes,
                    num_fed_loss_classes=self.fed_loss_num_classes,
                    num_classes=K,
                    weight=self.fed_loss_cls_weights,
                )
                fed_loss_classes_mask = fed_loss_classes.new_zeros(K + 1)
                fed_loss_classes_mask[fed_loss_classes] = 1
                fed_loss_classes_mask = fed_loss_classes_mask[:K]
                weight = fed_loss_classes_mask.view(1, K).expand(N, K).float()

                loss_ce += torch.sum(cls_loss * weight) / num_boxes
            else:
                loss_ce += torch.sum(cls_loss) / num_boxes

            losses = {'loss_ce': loss_ce}
        else:
            raise NotImplementedError

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes']

        batch_size = len(targets)
        pred_box_list = []
        pred_norm_box_list = []
        tgt_box_list = []
        tgt_box_xyxy_list = []
        for batch_idx in range(batch_size):
            valid_query = indices[batch_idx%(batch_size//2)][0]
            gt_multi_idx = indices[batch_idx%(batch_size//2)][1]
            if len(gt_multi_idx) == 0:
                continue
            bz_image_whwh = targets[batch_idx]['image_size_xyxy']
            bz_src_boxes = src_boxes[batch_idx]
            bz_target_boxes = targets[batch_idx]["boxes"]  # normalized (cx, cy, w, h)
            bz_target_boxes_xyxy = targets[batch_idx]["boxes_xyxy"]  # absolute (x1, y1, x2, y2)
            pred_box_list.append(bz_src_boxes[valid_query])
            pred_norm_box_list.append(bz_src_boxes[valid_query] / bz_image_whwh)  # normalize (x1, y1, x2, y2)
            tgt_box_list.append(bz_target_boxes[gt_multi_idx])
            tgt_box_xyxy_list.append(bz_target_boxes_xyxy[gt_multi_idx])

        if len(pred_box_list) != 0:
            src_boxes = torch.cat(pred_box_list)
            src_boxes_re=torch.cat(pred_box_list[-len(pred_box_list)//2:]+pred_box_list[:len(pred_box_list)//2])
            src_boxes_norm = torch.cat(pred_norm_box_list)  # normalized (x1, y1, x2, y2)
            target_boxes = torch.cat(tgt_box_list)
            target_boxes_abs_xyxy = torch.cat(tgt_box_xyxy_list)
            target_boxes_abs_xyxy_re=torch.cat(tgt_box_xyxy_list[-len(tgt_box_xyxy_list)//2:]+tgt_box_xyxy_list[:len(tgt_box_xyxy_list)//2])
            num_boxes = src_boxes.shape[0]
            losses = {}
            # require normalized (x1, y1, x2, y2)
            loss_bbox = F.l1_loss(src_boxes_norm, box_cxcywh_to_xyxy(target_boxes), reduction='none')
            losses['loss_bbox'] = loss_bbox.sum() / num_boxes

            # loss_giou = giou_loss(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
            loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes,src_boxes_re,target_boxes_abs_xyxy,target_boxes_abs_xyxy_re))
            losses['loss_giou'] = loss_giou.sum() / num_boxes
        else:
            losses = {'loss_bbox': outputs['pred_boxes'].sum() * 0,
                      'loss_giou': outputs['pred_boxes'].sum() * 0}

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices, _ = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)//2
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, _ = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class HungarianMatcherDynamicK(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-k (dynamic) matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,  cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, cost_mask: float = 1, use_focal: bool = False,use_fed_loss=False):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.use_fed_loss = use_fed_loss
        self.ota_k = 5
        if self.use_focal:
            self.focal_loss_alpha = 0.25
            self.focal_loss_gamma = 2.0
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0,  "all costs cant be 0"

    def forward(self, outputs, targets):
        """ simOTA for detr"""
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            conf_score=outputs["pred_scores"]
            # We flatten to compute the cost matrices in a batch
            pred_logits_pre,pred_logits_curr=torch.split(outputs["pred_logits"],bs//2,dim=0)
            out_bbox_pre,out_bbox_curr = torch.split(outputs["pred_boxes"],bs//2,dim=0)
            if self.use_focal or self.use_fed_loss:
                out_prob_pre = torch.sqrt(pred_logits_pre.sigmoid()*conf_score)  # [batch_size, num_queries, num_classes]
                out_prob_curr = torch.sqrt(pred_logits_curr.sigmoid()*conf_score)
            else:
                out_prob_pre = torch.sqrt(pred_logits_pre.softmax(-1)*conf_score) # [batch_size, num_queries, num_classes]
                out_prob_curr=torch.sqrt(pred_logits_curr.softmax(-1)*conf_score)
            indices = []
            matched_ids = []
            assert bs == len(targets)
            for batch_idx in range(bs//2):
                bz_boxes_pre = out_bbox_pre[batch_idx]  # [num_proposals, 4]
                bz_out_prob_pre = out_prob_pre[batch_idx]
                bz_boxes_curr = out_bbox_curr[batch_idx]  # [num_proposals, 4]
                bz_out_prob_curr = out_prob_curr[batch_idx]
                bz_tgt_ids_pre = targets[batch_idx]["labels"]
                bz_tgt_ids_curr = targets[batch_idx+bs//2]["labels"]
                num_insts = len(bz_tgt_ids_pre)
                assert len(bz_tgt_ids_curr)==num_insts,"YaHoo {},{}!".format(len(bz_tgt_ids_curr),num_insts)
                if num_insts == 0:  # empty object in key frame
                    non_valid = torch.zeros(bz_out_prob_pre.shape[0]).to(bz_out_prob_pre) > 0
                    indices_batchi = (non_valid, torch.arange(0, 0).to(bz_out_prob_pre))
                    matched_qidx = torch.arange(0, 0).to(bz_out_prob_pre)
                    indices.append(indices_batchi)
                    matched_ids.append(matched_qidx)
                    continue

                bz_gtboxs_pre = targets[batch_idx]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                bz_gtboxs_abs_xyxy_pre = targets[batch_idx]['boxes_xyxy']
                bz_gtboxs_curr = targets[batch_idx+bs//2]['boxes']  # [num_gt, 4] normalized (cx, xy, w, h)
                bz_gtboxs_abs_xyxy_curr = targets[batch_idx+bs//2]['boxes_xyxy']
                fg_mask_pre, is_in_boxes_and_center_pre = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes_pre),  # absolute (cx, cy, w, h)
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy_pre),  # absolute (cx, cy, w, h)
                    expanded_strides=32
                )
                fg_mask_curr, is_in_boxes_and_center_curr = self.get_in_boxes_info(
                    box_xyxy_to_cxcywh(bz_boxes_curr),  # absolute (cx, cy, w, h)
                    box_xyxy_to_cxcywh(bz_gtboxs_abs_xyxy_curr),  # absolute (cx, cy, w, h)
                    expanded_strides=32
                )
                fg_mask=fg_mask_pre&fg_mask_curr 
                is_in_boxes_and_center=is_in_boxes_and_center_pre&is_in_boxes_and_center_curr

                pair_wise_ious_pre = ops.box_iou(bz_boxes_pre, bz_gtboxs_abs_xyxy_pre)
                pair_wise_ious_curr = ops.box_iou(bz_boxes_curr, bz_gtboxs_abs_xyxy_curr)
                pair_wise_ious=(pair_wise_ious_pre+pair_wise_ious_curr)/2
                cost_class=0
                bz_out_prob_set=[bz_out_prob_pre,bz_out_prob_curr]
                bz_tgt_ids_set=[bz_tgt_ids_pre,bz_tgt_ids_curr]
                # Compute the classification cost.
                if self.use_focal:
                    alpha = self.focal_loss_alpha
                    gamma = self.focal_loss_gamma
                    for bz_out_prob,bz_tgt_ids in zip(bz_out_prob_set,bz_tgt_ids_set):
                        neg_cost_class = (1 - alpha) * (bz_out_prob ** gamma) * (-(1 - bz_out_prob + 1e-8).log())
                        pos_cost_class = alpha * ((1 - bz_out_prob) ** gamma) * (-(bz_out_prob + 1e-8).log())
                        cost_class += pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                elif self.use_fed_loss:
                    # focal loss degenerates to naive one
                    for bz_out_prob,bz_tgt_ids in zip(bz_out_prob_set,bz_tgt_ids_set):
                        neg_cost_class = (-(1 - bz_out_prob + 1e-8).log())
                        pos_cost_class = (-(bz_out_prob + 1e-8).log())
                        cost_class += pos_cost_class[:, bz_tgt_ids] - neg_cost_class[:, bz_tgt_ids]
                else:
                    for bz_out_prob,bz_tgt_ids in zip(bz_out_prob_set,bz_tgt_ids_set):
                        cost_class += -bz_out_prob[:, bz_tgt_ids]

                # Compute the L1 cost between boxes
                # image_size_out = torch.cat([v["image_size_xyxy"].unsqueeze(0) for v in targets])
                # image_size_out = image_size_out.unsqueeze(1).repeat(1, num_queries, 1).flatten(0, 1)
                # image_size_tgt = torch.cat([v["image_size_xyxy_tgt"] for v in targets])

                bz_image_size_out_pre = targets[batch_idx]['image_size_xyxy']
                bz_image_size_tgt_pre = targets[batch_idx]['image_size_xyxy_tgt']
                bz_image_size_out_curr = targets[batch_idx+bs//2]['image_size_xyxy']
                bz_image_size_tgt_curr = targets[batch_idx+bs//2]['image_size_xyxy_tgt']

                bz_out_bbox_pre = bz_boxes_pre / bz_image_size_out_pre  # normalize (x1, y1, x2, y2)
                bz_out_bbox_curr = bz_boxes_curr / bz_image_size_out_curr  # normalize (x1, y1, x2, y2)
                bz_tgt_bbox_pre = bz_gtboxs_abs_xyxy_pre / bz_image_size_tgt_pre  # normalize (x1, y1, x2, y2)
                bz_tgt_bbox_curr = bz_gtboxs_abs_xyxy_curr / bz_image_size_tgt_curr  # normalize (x1, y1, x2, y2)
                cost_bbox_pre = torch.cdist(bz_out_bbox_pre, bz_tgt_bbox_pre, p=1)
                cost_bbox_curr = torch.cdist(bz_out_bbox_curr, bz_tgt_bbox_curr, p=1)

                cost_giou = -generalized_box_iou(bz_boxes_pre,bz_boxes_curr,bz_gtboxs_abs_xyxy_pre,bz_gtboxs_abs_xyxy_curr)

                # Final cost matrix
                cost = self.cost_bbox * (cost_bbox_pre+cost_bbox_curr)/2 + self.cost_class * cost_class/2 + self.cost_giou * cost_giou + 100.0 * (~is_in_boxes_and_center)
                assert not torch.any(torch.isnan(cost)),"Error nan value occurs"
                # cost = (cost_class + 3.0 * cost_giou + 100.0 * (~is_in_boxes_and_center))  # [num_query,num_gt]
                cost[~fg_mask] = cost[~fg_mask] + 10000.0

                # if bz_gtboxs.shape[0]>0:
                indices_batchi, matched_qidx = self.dynamic_k_matching(cost, pair_wise_ious, bz_gtboxs_pre.shape[0])

                indices.append(indices_batchi)
                matched_ids.append(matched_qidx)

        return indices, matched_ids

    def get_in_boxes_info(self, boxes, target_gts, expanded_strides):
        xy_target_gts = box_cxcywh_to_xyxy(target_gts)  # (x1, y1, x2, y2)

        anchor_center_x = boxes[:, 0].unsqueeze(1)
        anchor_center_y = boxes[:, 1].unsqueeze(1)

        # whether the center of each anchor is inside a gt box
        b_l = anchor_center_x > xy_target_gts[:, 0].unsqueeze(0)
        b_r = anchor_center_x < xy_target_gts[:, 2].unsqueeze(0)
        b_t = anchor_center_y > xy_target_gts[:, 1].unsqueeze(0)
        b_b = anchor_center_y < xy_target_gts[:, 3].unsqueeze(0)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]
        # in fixed center
        center_radius = 2.5
        # Modified to self-adapted sampling --- the center size depends on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212
        b_l = anchor_center_x > (target_gts[:, 0] - (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_r = anchor_center_x < (target_gts[:, 0] + (center_radius * (xy_target_gts[:, 2] - xy_target_gts[:, 0]))).unsqueeze(0)
        b_t = anchor_center_y > (target_gts[:, 1] - (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)
        b_b = anchor_center_y < (target_gts[:, 1] + (center_radius * (xy_target_gts[:, 3] - xy_target_gts[:, 1]))).unsqueeze(0)

        is_in_centers = ((b_l.long() + b_r.long() + b_t.long() + b_b.long()) == 4)
        is_in_centers_all = is_in_centers.sum(1) > 0

        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)

        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, num_gt):
        matching_matrix = torch.zeros_like(cost)  # [300,num_gt]
        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = self.ota_k

        # Take the sum of the predicted value and the top 10 iou of gt with the largest iou as dynamic_k
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(cost[:, gt_idx], k=dynamic_ks[gt_idx].item(), largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(1)

        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[anchor_matching_gt > 1], dim=1)
            matching_matrix[anchor_matching_gt > 1] *= 0
            matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1

        while (matching_matrix.sum(0) == 0).any():
            num_zero_gt = (matching_matrix.sum(0) == 0).sum()
            matched_query_id = matching_matrix.sum(1) > 0
            cost[matched_query_id] += 100000.0
            unmatch_id = torch.nonzero(matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            if (matching_matrix.sum(1) > 1).sum() > 0:  # If a query matches more than one gt
                _, cost_argmin = torch.min(cost[anchor_matching_gt > 1],
                                           dim=1)  # find gt for these queries with minimal cost
                matching_matrix[anchor_matching_gt > 1] *= 0  # reset mapping relationship
                matching_matrix[anchor_matching_gt > 1, cost_argmin,] = 1  # keep gt with minimal cost

        assert not (matching_matrix.sum(0) == 0).any()
        selected_query = matching_matrix.sum(1) > 0
        gt_indices = matching_matrix[selected_query].max(1)[1]
        assert selected_query.sum() == len(gt_indices)

        cost[matching_matrix == 0] = cost[matching_matrix == 0] + float('inf')
        matched_query_id = torch.min(cost, dim=0)[1]

        return (selected_query, gt_indices), matched_query_id
