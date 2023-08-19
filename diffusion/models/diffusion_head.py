import math
import random
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import nms,box_iou

from .diffusion_losses import SetCriterionDynamicK, HungarianMatcherDynamicK
from .diffusion_models import DynamicHead

from yolox.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from yolox.utils import synchronize
from detectron2.layers import batched_nms
import time

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class DiffusionHead(nn.Module):
    """
    Implement DiffusionHead
    """

    def __init__(self,
                num_classes,
                width=1.0,
                strides=[8, 16, 32],
                num_proposals=500,
                num_heads=6,):
        super().__init__()
        self.device="cpu"
        self.dtype=torch.float32
        self.width=width
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        # self.num_proposals = 512
        self.hidden_dim = int(256*width)
        self.num_heads = num_heads

        # build diffusion
        timesteps = 1000
        sampling_timesteps = 1
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # tracking setting
        self.inference_time_range=1
        self.track_candidate=1
        self.candidate_num_strategy=max

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = 2.0
        self.box_renewal = True
        self.use_ensemble = True

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # Build Dynamic Head.
        class_weight = 2.0
        giou_weight = 2.0
        l1_weight = 5.0
        no_object_weight =0.1
        self.deep_supervision = True
        self.use_focal = True
        self.use_fed_loss = False
        self.use_nms = False
        self.pooler_resolution=7
        self.noise_strategy="xywh"
   
        self.head = DynamicHead(num_classes,self.hidden_dim,self.pooler_resolution,strides,[self.hidden_dim]*len(strides),return_intermediate=self.deep_supervision,num_heads=self.num_heads,use_focal=self.use_focal,use_fed_loss=self.use_fed_loss)
        # Loss parameters:

        # Build Criterion.
        matcher = HungarianMatcherDynamicK(
            cost_class=class_weight, cost_bbox=l1_weight, cost_giou=giou_weight, use_focal=self.use_focal,use_fed_loss=self.use_fed_loss
        )
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}
        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight,
            losses=losses, use_focal=self.use_focal,use_fed_loss=self.use_fed_loss)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, backbone_feats,images_whwh,x,t,lost_features=None,fix_bboxes=False,x_self_cond=None,clip_x_start=False):

        def prepare(x,images_whwh):
            x_boxes = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x_boxes = ((x_boxes / self.scale) + 1) / 2
            x_boxes = box_cxcywh_to_xyxy(x_boxes)
            x_boxes = x_boxes * images_whwh[:, None, :]
            return x_boxes
        
        def post(x_start,images_whwh):
            x_start = x_start / images_whwh[:, None, :]
            x_start = box_xyxy_to_cxcywh(x_start)
            x_start = (x_start * 2 - 1.) * self.scale
            x_start = torch.clamp(x_start, min=-1 * self.scale, max=self.scale)
            return x_start
        
        bs=len(x)//2
        bboxes=prepare(x,images_whwh=images_whwh)
        start_time=time.time()
        outputs_class, outputs_coord,outputs_score = self.head(backbone_feats,torch.split(bboxes,bs,dim=0),t,lost_features,fix_bboxes)
        end_time=time.time()

        x_start = outputs_coord[-1]  # (batch, num_proposals, 4) predict boxes: absolute coordinates (x1, y1, x2, y2)
        x_start=post(x_start,images_whwh=images_whwh)
        pred_noise = self.predict_noise_from_start(x,t,x_start)
        return ModelPrediction(pred_noise, x_start), outputs_class,outputs_coord,outputs_score,end_time-start_time
    
    @torch.no_grad()
    def new_ddim_sample(self,backbone_feats,images_whwh,ref_targets=None,dynamic_time=True,num_timesteps=1,num_proposals=500,inference_time_range=1,track_candidate=1,diffusion_t=200,clip_denoised=True):
        batch = images_whwh.shape[0]//2
        self.sampling_timesteps,self.num_proposals,self.track_candidate,self.inference_time_range=num_timesteps,num_proposals,track_candidate,inference_time_range
        shape = (batch, self.num_proposals, 4)
        cur_bboxes= torch.randn(shape,device=self.device,dtype=self.dtype)
        ref_t_list=[]
        track_t_list=[]
        total_time=0
        if ref_targets is None or self.track_candidate==0:
            ref_bboxes=torch.randn(shape, device=self.device)
            for i in range(batch):
                t = torch.randint(self.num_timesteps-self.inference_time_range, self.num_timesteps,(2,), device=self.device).long()
                if dynamic_time:
                    ref_t,track_t=t[0],t[1]
                else:
                    ref_t,track_t=t[0],t[0]
                ref_t_list.append(ref_t)
                track_t_list.append(track_t)
        else:
            labels =ref_targets[..., :5]
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
            shape = (batch, self.num_proposals, 4)
            diffused_boxes = []
            cur_diffused_boxes=[]
            for batch_idx,num_gt in enumerate(nlabel):
                gt_bboxes_per_image = box_cxcywh_to_xyxy(labels[batch_idx, :num_gt])
                image_size_xyxy = images_whwh[batch_idx]
                gt_boxes = gt_bboxes_per_image  / image_size_xyxy
                # cxcywh
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                # t = torch.randint(self.num_timesteps-self.inference_time_range, self.num_timesteps,(2,), device=self.device).long()
                # if dynamic_time:
                #     ref_t,track_t=t[0],t[1]
                # else:
                #     ref_t,track_t=t[0],t[0]
                if batch_idx==0:
                    ref_t=diffusion_t
                    track_t=diffusion_t
                else:
                    ref_t=diffusion_t
                    track_t=diffusion_t
                    self.track_candidate=4
                d_boxes,d_noise,ref_label= self.prepare_diffusion_concat(gt_boxes,ref_t)
                diffused_boxes.append(d_boxes)
                ref_t_list.append(ref_t)
                d_boxes,d_noise,ref_label= self.prepare_diffusion_concat(gt_boxes,track_t,ref_label)
                cur_diffused_boxes.append(d_boxes)
                track_t_list.append(track_t)
            ref_bboxes=torch.stack(diffused_boxes)
            cur_bboxes=torch.stack(cur_diffused_boxes)


        sampling_timesteps, eta= self.sampling_timesteps, self.ddim_sampling_eta

        def get_time_pairs(t,sampling_timesteps):
            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.linspace(-1, t - 1, steps=sampling_timesteps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            return time_pairs
        
        ref_t_time_pairs_list=torch.tensor([get_time_pairs(t,sampling_timesteps) for t in ref_t_list],device=self.device,dtype=torch.long)
        track_t_time_pairs_list=torch.tensor([get_time_pairs(t,sampling_timesteps) for t in track_t_list],device=self.device,dtype=torch.long)
        # (batch,sampling_timesteps,2)
        bboxes=torch.cat([ref_bboxes,cur_bboxes],dim=0)

        x_start = None
        # for (ref_time, ref_time_next),(cur_time, cur_time_next) in zip(ref_time_pairs,cur_time_pairs):
        for sampling_timestep in range(sampling_timesteps):
            is_last=sampling_timestep==(sampling_timesteps-1)

            ref_time_cond = ref_t_time_pairs_list[:,sampling_timestep,0]
            cur_time_cond = track_t_time_pairs_list[:,sampling_timestep,0]

            time_cond=torch.cat([ref_time_cond,cur_time_cond],dim=0)

            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord,outputs_score,association_time = self.model_predictions(backbone_feats,images_whwh,bboxes,time_cond,fix_bboxes=False,
                                                                         x_self_cond=self_cond, clip_x_start=clip_denoised)
            total_time+=association_time
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start
                

            if is_last:
                bboxes = x_start
                continue

            if self.box_renewal:  # filter
                remain_list=[]
                pre_remain_bboxes=[]
                pre_remain_x_start=[]
                pre_remain_pred_noise=[]
                cur_remain_bboxes=[]
                cur_remain_x_start=[]
                cur_remain_pred_noise=[]
                for i in range(batch):
                    # if i==0:
                    #     remain_list.append(len(pred_noise[i,:,:]))
                    #     pre_remain_pred_noise.append(pred_noise[i,:,:])
                    #     cur_remain_pred_noise.append(pred_noise[i+batch,:,:])
                    #     pre_remain_x_start.append(x_start[i,:,:])
                    #     cur_remain_x_start.append(x_start[i+batch,:,:])
                    #     pre_remain_bboxes.append(bboxes[i,:,:])
                    #     cur_remain_bboxes.append(bboxes[i+batch,:,:])
                    # else:
                    threshold = 0.2
                    score_per_image = outputs_score[-1][i]
                    # pre_score=torch.sqrt(score_per_image*torch.sigmoid(outputs_class[-1][i]))
                    # cur_score=torch.sqrt(score_per_image*torch.sigmoid(outputs_class[-1][i+batch]))
                    # value=((pre_score+cur_score)/2).flatten()
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                    keep_idx = value >=threshold
                    num_remain = torch.sum(keep_idx)
                    remain_list.append(num_remain)
                    pre_remain_pred_noise.append(pred_noise[i,keep_idx,:])
                    cur_remain_pred_noise.append(pred_noise[i+batch,keep_idx,:])
                    pre_remain_x_start.append(x_start[i,keep_idx,:])
                    cur_remain_x_start.append(x_start[i+batch,keep_idx,:])
                    pre_remain_bboxes.append(bboxes[i,keep_idx,:])
                    cur_remain_bboxes.append(bboxes[i+batch,keep_idx,:])
                x_start=pre_remain_x_start+cur_remain_x_start
                bboxes=pre_remain_bboxes+cur_remain_bboxes
                pred_noise=pre_remain_pred_noise+cur_remain_pred_noise

            def diffusion(sampling_times,bboxes,x_start,pred_noise):
                
                times,time_nexts=sampling_times[:,0],sampling_times[:,1]

                alpha = torch.tensor([self.alphas_cumprod[time] for time in times],dtype=self.dtype,device=self.device)
                alpha_next = torch.tensor([self.alphas_cumprod[time_next] for time_next in time_nexts],dtype=self.dtype,device=self.device)

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                if self.box_renewal:
                    for i in range(batch):
                        noise = torch.randn_like(bboxes[i])
                        bboxes[i] = x_start[i] * alpha_next[i].sqrt() + \
                            c[i] * pred_noise[i] + \
                            sigma[i] * noise
                        
                        bboxes[i] = torch.cat((bboxes[i], torch.randn(self.num_proposals - remain_list[i], 4, device=self.device)), dim=0)
                else:
                    noise = torch.randn_like(bboxes)

                    bboxes = x_start * alpha_next.sqrt()[:,None,None] + \
                        c[:,None,None] * pred_noise + \
                        sigma[:,None,None] * noise
                
                return bboxes
            
            bboxes[:batch]=diffusion(ref_t_time_pairs_list[:,sampling_timestep],bboxes[:batch],x_start[:batch],pred_noise[:batch])
            bboxes[batch:]=diffusion(track_t_time_pairs_list[:,sampling_timestep],bboxes[batch:],x_start[batch:],pred_noise[batch:])

            if self.box_renewal:
                bboxes=torch.stack(bboxes)

        box_cls = outputs_class[-1]
        box_pred = outputs_coord[-1]
        conf_score=outputs_score[-1]

        return torch.cat([box_pred.view(2*batch,-1,4),box_cls.view(2*batch,-1,1)],dim=-1),conf_score.view(batch,-1,1),total_time
    
    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward(self,features,mate_info,targets=None):

        mate_shape,mate_device,mate_dtype=mate_info
        self.device=mate_device
        self.dtype=mate_dtype
        b,_,h,w=mate_shape
        
        images_whwh=torch.tensor([w, h, w, h], dtype=self.dtype, device=self.device)[None,:].expand(2*b,4)
        if not self.training:
            results = self.new_ddim_sample(features,images_whwh,targets,dynamic_time=False)
            return results

        if self.training:
            targets, x_boxes, noises, t = self.prepare_targets(targets,images_whwh)
            t=t.squeeze(-1)
            # t[b:]=t[:b]
            x_boxes = x_boxes * images_whwh[:,None,:]
            pre_x_boxes,cur_x_boxes=torch.split(x_boxes,b,dim=0)

            outputs_class,outputs_coord,outputs_score = self.head(features,(pre_x_boxes,cur_x_boxes),t)
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1],'pred_scores':outputs_score[-1]}

            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b,'pred_scores': c}
                                         for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1],outputs_score[:-1])]
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict: 
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
 
    def prepare_diffusion_repeat(self,gt_boxes,t,ref_repeat_tensor=None):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.full((1,),t,device=self.device).long()

        noise = torch.randn(self.num_proposals,4,device=self.device,dtype=self.dtype)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=self.dtype, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)
        if ref_repeat_tensor is not None:
            repeat_tensor=ref_repeat_tensor

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        if self.noise_strategy=="xy":
            noise[:,2:]=0
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.training:
            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.

            diff_boxes = box_cxcywh_to_xyxy(x)
        else:
            diff_boxes=x

        return diff_boxes,noise,repeat_tensor

    def prepare_diffusion_concat(self,gt_boxes,t,ref_mask=None):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        if self.training:
            self.track_candidate=1
        t = torch.full((1,),t,device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device,dtype=self.dtype)
        select_mask=None
        num_gt = gt_boxes.shape[0]*self.track_candidate
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=self.dtype, device=self.device)
            num_gt = 1
        else:
            gt_boxes=torch.repeat_interleave(gt_boxes,torch.tensor([self.track_candidate]*gt_boxes.shape[0],device=self.device),dim=0)
        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device,dtype=self.dtype) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            # box_placeholder=torch.clip(torch.poisson(torch.clip(box_placeholder*5,min=0)),min=1,max=10)/10
            # box_placeholder=torch.nn.init.uniform_(box_placeholder, a=0, b=1)
            # box_placeholder=torch.ones_like(box_placeholder)
            # box_placeholder[:,:2]=box_placeholder[:,:2]/2
            box_placeholder[:, 2:4] = torch.clip(box_placeholder[:, 2:4], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            if ref_mask is not None:
                select_mask=ref_mask
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        if self.noise_strategy=="xy":
            noise[:,2:]=0
        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        if self.training:
            # x=x_start

            x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
            x = ((x / self.scale) + 1) / 2.

            diff_boxes = box_cxcywh_to_xyxy(x)
        else:
            diff_boxes = x

        return diff_boxes, noise, select_mask

    def prepare_targets(self,targets,images_whwh):
        labels = targets[..., :5]
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects
        new_targets = []
        diffused_boxes = []
        noises = []
        ts = []
        select_mask={}
        # select_t={}
        # select_gt_boxes={}
        for batch_idx,num_gt in enumerate(nlabel):
            target = {}
            gt_bboxes_per_image = box_cxcywh_to_xyxy(labels[batch_idx, :num_gt, 1:5])
            gt_classes = labels[batch_idx, :num_gt, 0]
            image_size_xyxy = images_whwh[batch_idx]
            gt_boxes = gt_bboxes_per_image  / image_size_xyxy
            # cxcywh
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            x_gt_boxes=gt_boxes
            d_t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()[0]
            ## baseline setting
            # if batch_idx<len(nlabel)//2:
            #     d_t = torch.randint(0, 40, (1,), device=self.device).long()[0]
            # else:
            #     d_t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()[0]
            # if select_t.get(batch_idx%(len(nlabel)//2),None) is not None:
            #     d_t=select_t.get(batch_idx%(len(nlabel)//2),None)
            # if select_gt_boxes.get(batch_idx%(len(nlabel)//2),None) is not None:
            #     x_gt_boxes=select_gt_boxes.get(batch_idx%(len(nlabel)//2),None)    
            d_boxes,d_noise,d_mask= self.prepare_diffusion_concat(x_gt_boxes,d_t,select_mask.get(batch_idx%(len(nlabel)//2),None))
            if d_mask is not None:
                select_mask[batch_idx%(len(nlabel)//2)]=d_mask
            # if d_t is not None:
            #     select_t[batch_idx%(len(nlabel)//2)]=d_t
            # if select_gt_boxes.get(batch_idx%(len(nlabel)//2),None) is None:
            #     select_gt_boxes[batch_idx%(len(nlabel)//2)]=gt_boxes 
            diffused_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            target["labels"] = gt_classes.long()
            target["boxes"] = gt_boxes
            target["boxes_xyxy"] = gt_bboxes_per_image
            target["image_size_xyxy"] = image_size_xyxy
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt
            new_targets.append(target)

        return new_targets, torch.stack(diffused_boxes), torch.stack(noises), torch.stack(ts)



