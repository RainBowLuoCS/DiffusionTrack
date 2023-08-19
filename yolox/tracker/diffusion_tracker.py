import numpy as np
from collections import deque

import torch
import torch.nn.functional as F 
import torchvision
from copy import deepcopy
from yolox.tracker import matching
from detectron2.structures import Boxes
from yolox.utils.box_ops import box_xyxy_to_cxcywh
from yolox.utils.boxes import xyxy2cxcywh
from torchvision.ops import box_iou,nms
from yolox.utils.cluster_nms import cluster_nms



class DiffusionTracker(object):
    def __init__(self,model,tensor_type,conf_thresh=0.7,det_thresh=0.6,nms_thresh_3d=0.7,nms_thresh_2d=0.75,interval=5):

        self.frame_id = 0
        self.backbone=model.backbone
        self.feature_projs=model.projs
        self.diffusion_model=model.head
        self.feature_extractor=self.diffusion_model.head.box_pooler
        self.det_thresh = det_thresh
        self.association_thresh = conf_thresh
        self.low_det_thresh = 0.1
        self.low_association_thresh = 0.2
        self.nms_thresh_2d=nms_thresh_2d
        self.nms_thresh_3d=nms_thresh_3d
        self.same_thresh=0.9
        self.pre_features=None
        self.data_type=tensor_type

        self.re_association_features=None
        self.re_association_interval=interval
        # [tracklet_id,T,6] (x,y,x,y,score,t)
        self.tracklet_db=None
        self.total_time=0
        self.dynamic_time=True
        self.repeat_times=8
        self.sampling_steps=1
        self.num_boxes=1000

        self.track_t=40
        self.re_association_t=40
        self.mot17=False

    def update(self,cur_image):
        self.frame_id += 1
        cur_features,mate_info=self.extract_feature(cur_image=cur_image)
        mate_shape,mate_device,mate_dtype=mate_info
        self.diffusion_model.device=mate_device
        self.diffusion_model.dtype=mate_dtype
        b,_,h,w=mate_shape
        images_whwh=torch.tensor([w, h, w, h], dtype=mate_dtype, device=mate_device)[None,:].expand(4*b,4)
        if self.frame_id==1:
            if self.pre_features is None:
                self.pre_features=cur_features
            inps=self.prepare_input(self.pre_features,cur_features)
            diffusion_outputs,conf_scores,association_time=self.diffusion_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                dynamic_time=self.dynamic_time,track_candidate=self.repeat_times)
            self.total_time+=association_time
            _,_,detections=self.diffusion_postprocess(diffusion_outputs,conf_scores,conf_thre=self.association_thresh,nms_thre=self.nms_thresh_3d)
            detections=self.diffusion_det_filt(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)
            self.tracklet_db=np.zeros((len(detections),1,6))
            self.tracklet_db[:,-1,:4]=detections[:,:4]
            self.tracklet_db[:,-1,4]=detections[:,5]
            self.tracklet_db[:,-1,5]=self.frame_id
        else:
            ref_bboxes,ref_track_ids=self.get_targets_from_tracklet_db()
            inps=self.prepare_input(self.pre_features,cur_features)
            bboxes=box_xyxy_to_cxcywh(torch.tensor(np.array(ref_bboxes))).type(self.data_type).reshape(1,-1,4).repeat(2,1,1)
            # ref_num_proposals=self.proposal_schedule(len(ref_bboxes))
            # ref_sampling_steps=self.sampling_steps_schedule(len(ref_bboxes))
            track_tracklet_db=np.concatenate([np.zeros((len(self.tracklet_db),1,5)),deepcopy(self.tracklet_db[:,-1,5]).reshape(-1,1,1)],axis=2)
            diffusion_outputs,conf_scores,association_time=self.diffusion_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                ref_targets=bboxes,dynamic_time=self.dynamic_time,track_candidate=self.repeat_times,diffusion_t=self.track_t)
            self.total_time+=association_time
            diffusion_ref_detections,diffusion_track_detections,detections=self.diffusion_postprocess(diffusion_outputs,
                                                                                                      conf_scores,
                                                                                                      conf_thre=self.low_association_thresh,
                                                                                                      nms_thre=self.nms_thresh_3d)
            high_track_inds=diffusion_ref_detections[:,4]>self.association_thresh
            diffusion_ref_detections,diffusion_track_detections=diffusion_ref_detections[high_track_inds],diffusion_track_detections[high_track_inds]
            
            detections=self.diffusion_det_filt(detections,conf_thre=self.low_det_thresh,nms_thre=self.nms_thresh_2d)
            diffusion_ref_detections,diffusion_track_detections=self.diffusion_track_filt(diffusion_ref_detections,
                                                                                          diffusion_track_detections,
                                                                                          conf_thre=self.low_det_thresh,
                                                                                          nms_thre=self.nms_thresh_2d)
            
            pred_track_ids,pred_bboxes,pred_scores=self.diffusion_matching(ref_bboxes,ref_track_ids,
                                                                           diffusion_ref_detections,
                                                                           diffusion_track_detections)
            
            high_det_inds=detections[:,5]>self.det_thresh

            if pred_bboxes is None:
                new_detections=detections
                new_detections_inds=high_det_inds
            else:
                dists = matching.iou_distance(pred_bboxes, detections[:,:4])
                if self.mot17:
                    dists=matching.fuse_score(dists,detections[:,5])
                matches,u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)
                new_detections=detections[u_detection]
                new_detections_inds=high_det_inds[u_detection]
                if len(matches)>0:
                    pred_bboxes[matches[:,0]]=detections[matches[:,1],:4]
            if ref_track_ids is not None and pred_track_ids is not None:
                matching_index=np.argwhere(np.array(ref_track_ids).reshape(-1,1)==pred_track_ids.reshape(1,-1))
                track_tracklet_db[ref_track_ids[matching_index[:,0]],-1,:4]=pred_bboxes[matching_index[:,1]]
                track_tracklet_db[ref_track_ids[matching_index[:,0]],-1,4]=pred_scores[matching_index[:,1]]
                track_tracklet_db[ref_track_ids[matching_index[:,0]],-1,5]=self.frame_id
                # self.track_t=400
                self.track_t=self.extract_mean_track_t(self.tracklet_db[ref_track_ids[matching_index[:,0]],-1,:4],pred_bboxes[matching_index[:,1]])
                # print(self.track_t)
            self.tracklet_db=np.concatenate([self.tracklet_db,track_tracklet_db],axis=1)
            # yolox init new tracks
            if len(new_detections[new_detections_inds])>0:
                new_detections=new_detections[new_detections_inds]
                pred_bboxes,pred_scores=new_detections[:,:4],new_detections[:,5]
                new_tracklet_db=np.zeros((len(new_detections),self.frame_id,6))
                new_tracklet_db[:,-1,:4]=pred_bboxes
                new_tracklet_db[:,-1,4]=pred_scores
                new_tracklet_db[:,-1,5]=self.frame_id
                self.tracklet_db=np.concatenate([self.tracklet_db,new_tracklet_db],axis=0)
                
        self.pre_features=cur_features
        if (self.frame_id-1)%self.re_association_interval==0:
            if self.frame_id!=1:
                # reassociation
                inps=self.prepare_input(self.re_association_features,cur_features)
                # images_whwh=torch.tensor([w, h, w, h], dtype=mate_dtype, device=mate_device)[None,:].expand(4*b,4)

                ref_mask=self.tracklet_db[:,-1-self.re_association_interval,:5].sum(-1)>0
                ref_bbox=deepcopy(self.tracklet_db[ref_mask,-1-self.re_association_interval,:4])
                ref_track_ids=np.arange(len(self.tracklet_db))[ref_mask]

                
                cur_mask=self.tracklet_db[:,-1,:5].sum(-1)>0
                cur_bbox=deepcopy(self.tracklet_db[cur_mask,-1,:4])
                cur_track_ids=np.arange(len(self.tracklet_db))[cur_mask]

                mix_mask=np.logical_and(ref_mask,cur_mask)
                if sum(mix_mask)>0:
                    # self.re_association_t=400
                    self.re_association_t=self.extract_mean_track_t(self.tracklet_db[mix_mask,-1-self.re_association_interval,:4],self.tracklet_db[mix_mask,-1,:4])

                bboxes=box_xyxy_to_cxcywh(torch.tensor(np.array(ref_bbox))).type(self.data_type).reshape(1,-1,4).repeat(2,1,1)

                diffusion_outputs,conf_scores,association_time=self.diffusion_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                    ref_targets=bboxes,dynamic_time=self.dynamic_time,track_candidate=self.repeat_times,diffusion_t=self.re_association_t)
                # self.total_time+=association_time
                diffusion_ref_detections,diffusion_track_detections,_=self.diffusion_postprocess(diffusion_outputs,
                                                                                                 conf_scores,
                                                                                                 conf_thre=self.association_thresh,
                                                                                                 nms_thre=self.nms_thresh_3d)
                
                diffusion_ref_detections,diffusion_track_detections=self.diffusion_track_filt(diffusion_ref_detections,
                                                                                diffusion_track_detections,
                                                                                conf_thre=self.det_thresh,
                                                                                nms_thre=self.nms_thresh_2d)

                pred_track_ids,pred_bboxes,pred_scores=self.diffusion_matching(ref_bbox,ref_track_ids,
                                                                diffusion_ref_detections,
                                                                diffusion_track_detections)
                if pred_bboxes is not None:
                    dists = matching.iou_distance(pred_bboxes,cur_bbox)
                    matches,u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)
                    if len(matches)>0:
                        re_aasociation_mask=pred_track_ids[matches[:,0]]!=cur_track_ids[matches[:,1]]
                        for pre_track_id,cur_track_id in zip(pred_track_ids[matches[:,0]][re_aasociation_mask],
                                                            cur_track_ids[matches[:,1]][re_aasociation_mask]):
                            if self.tracklet_db[cur_track_id,-1-self.re_association_interval,-1]==0 and pre_track_id!=cur_track_id and \
                                max(self.tracklet_db[pre_track_id,-1-self.re_association_interval:,-1])<max(self.tracklet_db[cur_track_id,-1-self.re_association_interval:,-1]):
                                self.tracklet_db[pre_track_id]=np.where(self.tracklet_db[pre_track_id]>self.tracklet_db[cur_track_id],self.tracklet_db[pre_track_id],self.tracklet_db[cur_track_id])
            self.re_association_features=cur_features

    def get_results(self):
        results=[]
        overall_obj_ids=np.arange(len(self.tracklet_db))
        for t in range(len(self.tracklet_db[0])):
            activated_mask=self.tracklet_db[:,t,:5].sum(-1)>0
            obj_info=self.tracklet_db[activated_mask,t,:]
            obj_track_ids=overall_obj_ids[activated_mask]
            results.append((obj_track_ids,obj_info))
        return results
    
    def extract_feature(self,cur_image):
        fpn_outs=self.backbone(cur_image)
        cur_features=[]
        for proj,l_feat in zip(self.feature_projs,fpn_outs):
            cur_features.append(proj(l_feat))
        mate_info=(cur_image.shape,cur_image.device,cur_image.dtype)
        return cur_features,mate_info

    def extract_mean_track_t(self,pre_box,cur_box):
        # "xyxy"
        pre_box=xyxy2cxcywh(pre_box)
        cur_box=xyxy2cxcywh(cur_box)
        abs_box=np.abs(pre_box-cur_box)
        abs_percent=np.sum(abs_box/(pre_box+1e-5),axis=1)/4
        track_t=np.mean(abs_percent)
        return min(max(int(track_t*1000),1),999)

    
    def diffusion_postprocess(self,diffusion_outputs,conf_scores,nms_thre=0.7,conf_thre=0.6):

        pre_prediction,cur_prediction=diffusion_outputs.split(len(diffusion_outputs)//2,dim=0)

        output = [None for _ in range(len(pre_prediction))]
        for i,(pre_image_pred,cur_image_pred,association_score) in enumerate(zip(pre_prediction,cur_prediction,conf_scores)):

            association_score=association_score.flatten()
            # If none are remaining => process next image
            if not pre_image_pred.size(0):
                continue
            # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections=torch.zeros((2,len(cur_image_pred),7),dtype=cur_image_pred.dtype,device=cur_image_pred.device)
            detections[0,:,:4]=pre_image_pred[:,:4]
            detections[1,:,:4]=cur_image_pred[:,:4]
            detections[0,:,4]=association_score
            detections[1,:,4]=association_score
            detections[0,:,5]=torch.sqrt(torch.sigmoid(pre_image_pred[:,4])*association_score)
            detections[1,:,5]=torch.sqrt(torch.sigmoid(cur_image_pred[:,4])*association_score)

            score_out_index=association_score>conf_thre

            # strategy=torch.mean
            # value=strategy(detections[:,:,5],dim=0,keepdim=False)
            # score_out_index=value>conf_thre

            detections=detections[:,score_out_index,:]

            if not detections.size(1):
                output[i]=detections
                continue

            nms_out_index_3d = cluster_nms(
                                        detections[0,:,:4],
                                        detections[1,:,:4],
                                        # value[score_out_index],
                                        detections[0,:,4],
                                        iou_threshold=nms_thre)

            detections = detections[:,nms_out_index_3d,:]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))

        return output[0][0],output[0][1],torch.cat([output[1][0],output[1][1]],dim=0) if len(output)>=2 else None

    def diffusion_track_filt(self,ref_detections,track_detections,conf_thre=0.6,nms_thre=0.7):

        if not ref_detections.size(1):
            return ref_detections.cpu().numpy(),track_detections.cpu().numpy()
        
        scores=ref_detections[:,5]
        score_out_index=scores>conf_thre
        ref_detections=ref_detections[score_out_index]
        track_detections=track_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
                ref_detections[:, :4],
                ref_detections[:, 5],
                ref_detections[:, 6],
                nms_thre,
            )
        return ref_detections[nms_out_index].cpu().numpy(),track_detections[nms_out_index].cpu().numpy()

    def diffusion_det_filt(self,diffusion_detections,conf_thre=0.6,nms_thre=0.7):

        if not diffusion_detections.size(1):
            return diffusion_detections.cpu().numpy()

        scores=diffusion_detections[:,5]
        score_out_index=scores>conf_thre
        diffusion_detections=diffusion_detections[score_out_index]
        nms_out_index = torchvision.ops.batched_nms(
                diffusion_detections[:, :4],
                diffusion_detections[:, 5],
                diffusion_detections[:, 6],
                nms_thre,
            )
        return diffusion_detections[nms_out_index].cpu().numpy()
    
    def diffusion_matching(self,ref_bboxes,ref_track_ids,diffusion_pre_track_outputs,diffusion_cur_track_outputs):
        ref_bboxes=np.array(ref_bboxes)
        dists=matching.iou_distance(ref_bboxes,diffusion_pre_track_outputs[:,:4])
        matches,u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)
        if len(matches)>0:
            ref_track_ids=np.array(ref_track_ids)[matches[:,0]]
            return ref_track_ids,diffusion_cur_track_outputs[matches[:,1],:4],diffusion_cur_track_outputs[matches[:,1],5]
        else:
            return None,None,None
    
    def proposal_schedule(self,num_ref_bboxes):
        # simple strategy
        return 16*num_ref_bboxes
    
    def sampling_steps_schedule(self,num_ref_bboxes):
        min_sampling_steps=1
        max_sampling_steps=4
        min_num_bboxes=10
        max_num_bboxes=100
        ref_sampling_steps=(num_ref_bboxes-min_num_bboxes)*(max_sampling_steps-min_sampling_steps)/(max_num_bboxes-min_num_bboxes)+min_sampling_steps

        return min(max(int(ref_sampling_steps),min_sampling_steps),max_sampling_steps)

    def vote_to_remove_candidate(self,track_ids,detections,vote_iou_thres=0.75,sorted=False,descending=False):

        box_pred_per_image, scores_per_image=detections[:,:4],detections[:,4]*detections[:,5]
        score_track_indices=torch.argsort((track_ids+scores_per_image),descending=True)
        track_ids=track_ids[score_track_indices]
        scores_per_image=scores_per_image[score_track_indices]
        box_pred_per_image=box_pred_per_image[score_track_indices]

        assert len(track_ids)==box_pred_per_image.shape[0]

        # vote guarantee only one track id in track candidates
        keep_mask = torch.zeros_like(scores_per_image, dtype=torch.bool)
        for class_id in torch.unique(track_ids):
            curr_indices = torch.where(track_ids == class_id)[0]
            curr_keep_indices = nms(box_pred_per_image[curr_indices],scores_per_image[curr_indices],vote_iou_thres)
            candidate_iou_indices=box_iou(box_pred_per_image[curr_indices],box_pred_per_image[curr_indices])>vote_iou_thres
            counter=[]
            for cluster_indice in candidate_iou_indices[curr_keep_indices]:
                cluster_scores=scores_per_image[curr_indices][cluster_indice]
                counter.append(len(cluster_scores)+torch.mean(cluster_scores))
            max_indice=torch.argmax(torch.tensor(counter).type(self.data_type))
            keep_mask[curr_indices[curr_keep_indices][max_indice]] = True
        
        keep_indices = torch.where(keep_mask)[0]        
        track_ids=track_ids[keep_indices]
        box_pred_per_image=box_pred_per_image[keep_indices]
        scores_per_image=scores_per_image[keep_indices]

        if sorted and not descending:
            descending_indices=torch.argsort(track_ids)
            track_ids=track_ids[descending_indices]
            box_pred_per_image=box_pred_per_image[descending_indices]
            scores_per_image=scores_per_image[descending_indices]

        return track_ids.cpu().numpy(),box_pred_per_image.cpu().numpy(),scores_per_image.cpu().numpy()

    def prepare_input(self,pre_features,cur_features):
        inps_pre_features=[]
        inps_cur_Features=[]
        for l_pre_feat,l_cur_feat in zip(pre_features,cur_features):
            inps_pre_features.append(torch.cat([l_pre_feat.clone(),l_cur_feat.clone()],dim=0))
            inps_cur_Features.append(torch.cat([l_cur_feat.clone(),l_cur_feat.clone()],dim=0))
        return (inps_pre_features,inps_cur_Features)

    def get_targets_from_tracklet_db(self):
        ref_mask=self.tracklet_db[:,-1,:5].sum(-1)>0
        ref_bbox=deepcopy(self.tracklet_db[ref_mask,-1,:4])
        ref_track_ids=np.arange(len(self.tracklet_db))[ref_mask]
        return ref_bbox,ref_track_ids


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


