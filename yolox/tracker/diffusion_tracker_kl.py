import numpy as np
from collections import deque
import time
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

from .kalman_filter import KalmanFilter
from yolox.tracker import matching
from .basetrack import BaseTrack, TrackState

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self._tlwh=new_track.tlwh
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self._tlwh=new_tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)

class DiffusionTracker(object):
    def __init__(self,model,tensor_type,conf_thresh=0.7,det_thresh=0.6,nms_thresh_3d=0.7,nms_thresh_2d=0.75,interval=5,detections=None):

        self.frame_id = 0
        # BaseTrack._count=-1
        self.backbone=model.backbone
        self.feature_projs=model.projs
        self.diffusion_model=model.head
        self.feature_extractor=self.diffusion_model.head.box_pooler
        self.det_thresh = det_thresh
        self.association_thresh = conf_thresh
        # self.low_det_thresh = 0.1
        # self.low_association_thresh = 0.2
        self.nms_thresh_2d=nms_thresh_2d
        self.nms_thresh_3d=nms_thresh_3d
        self.same_thresh=0.9
        self.pre_features=None
        self.data_type=tensor_type
        self.detections=detections

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.max_time_lost = 30
        self.kalman_filter = KalmanFilter()

        self.repeat_times=0
        self.dynamic_time=True
        
        self.sampling_steps=1
        self.num_boxes=500

        self.track_t=400
        self.mot17=False

    def update(self,cur_image):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
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
            _,_,detections=self.diffusion_postprocess(diffusion_outputs,conf_scores,conf_thre=self.association_thresh,nms_thre=self.nms_thresh_3d)
            detections=self.diffusion_det_filt(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)
            # detections=np.array(self.detections[self.frame_id])
            # detections=detections[detections[:,5]>self.det_thresh]
            for det in detections:
                track=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                track.activate(self.kalman_filter, self.frame_id)
                self.tracked_stracks.append(track)
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]
            return output_stracks,association_time
        else:
            ref_bboxes=[STrack.tlwh_to_tlbr(track._tlwh) for track in self.tracked_stracks]
            inps=self.prepare_input(self.pre_features,cur_features)
            if len(ref_bboxes)>0:
                bboxes=box_xyxy_to_cxcywh(torch.tensor(np.array(ref_bboxes))).type(self.data_type).reshape(1,-1,4).repeat(2,1,1)
            else:
                bboxes=None
            # ref_num_proposals=self.proposal_schedule(len(ref_bboxes))
            # ref_sampling_steps=self.sampling_steps_schedule(len(ref_bboxes))
            diffusion_outputs,conf_scores,association_time=self.diffusion_model.new_ddim_sample(inps,images_whwh,num_timesteps=self.sampling_steps,num_proposals=self.num_boxes,
                                                                                                ref_targets=bboxes,dynamic_time=self.dynamic_time,track_candidate=self.repeat_times,diffusion_t=self.track_t)
            diffusion_ref_detections,diffusion_track_detections,detections=self.diffusion_postprocess(diffusion_outputs,
                                                                                                      conf_scores,
                                                                                                      conf_thre=self.association_thresh,
                                                                                                      nms_thre=self.nms_thresh_3d)
            
            detections=self.diffusion_det_filt(detections,conf_thre=self.det_thresh,nms_thre=self.nms_thresh_2d)
            # detections=np.array(self.detections[self.frame_id])
            # if len(detections)>0:
            #     detections=detections[detections[:,5]>self.det_thresh]
            diffusion_ref_detections,diffusion_track_detections=self.diffusion_track_filt(diffusion_ref_detections,
                                                                                          diffusion_track_detections,
                                                                                          conf_thre=self.det_thresh,
                                                                                          nms_thre=self.nms_thresh_2d)
            start_time=time.time()
            STrack.multi_predict(self.tracked_stracks)
            dists = matching.iou_distance(ref_bboxes, diffusion_ref_detections[:,:4])
            matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.same_thresh)

            if len(matches)>0:
                # fix position with detection result
                dists_fix=matching.iou_distance(diffusion_track_detections[matches[:,1],:4],detections[:,:4])
                matches_fix, u_track_fix, u_detection_fix = matching.linear_assignment(dists_fix, thresh=self.same_thresh)
                if len(matches_fix)>0:
                    diffusion_track_detections[matches[:,1]][matches_fix[:,0],:4]=detections[matches_fix[:,1],:4]

                # filt detection with tracked result
                detections=detections[u_detection_fix]

            ref_box_t=[]
            track_box_t=[]
            for itracked, idet in matches:
                track = self.tracked_stracks[itracked]
                ref_box_t.append(STrack.tlwh_to_tlbr(track._tlwh))
                det = diffusion_track_detections[idet]
                track_box_t.append(det[:4])
                new_strack=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                if track.state == TrackState.Tracked:
                    track.update(new_strack, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(new_strack, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            if len(ref_box_t)>0:
                self.track_t=self.extract_mean_track_t(np.array(ref_box_t),np.array(track_box_t))
            for it in u_track:
                track = self.tracked_stracks[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)
            
            STrack.multi_predict(self.lost_stracks)

            dists_lost = matching.iou_distance([track.tlbr for track in self.lost_stracks], detections[:4])
            matches_lost, u_track_lost, u_detection_lost = matching.linear_assignment(dists_lost, thresh=self.same_thresh)

            for itracked, idet in matches_lost:
                track = self.lost_stracks[itracked]
                det = detections[idet]
                new_strack=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                if track.state == TrackState.Tracked:
                    track.update(new_strack, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(new_strack, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            

            for inew in u_detection_lost:
            # for inew in range(len(detections)):
                det = detections[inew]
                track=STrack(STrack.tlbr_to_tlwh(det[:4]), det[5])
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)

            for track in self.lost_stracks:
                if self.frame_id - track.end_frame > self.max_time_lost:
                    track.mark_removed()
                    removed_stracks.append(track)
            

            self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
            self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
            self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
            self.lost_stracks.extend(lost_stracks)
            self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
            self.removed_stracks.extend(removed_stracks)
            self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
            # get scores of lost tracks
           

        self.pre_features=cur_features
        output_stracks = [track for track in self.tracked_stracks]
        return output_stracks,association_time+time.time()-start_time
    
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
        # min(max(int(track_t*1000),1),999)
        # min(max(int((np.exp(track_t)-1)/(np.exp(0)-1)*1000),1),999)
        # min(max(int(np.log(track_t+1)/np.log(2)*1000),1),999)
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

    # def get_targets_from_tracklet_db(self):
    #     ref_mask=self.tracklet_db[:,-1,:5].sum(-1)>0
    #     ref_bbox=deepcopy(self.tracklet_db[ref_mask,-1,:4])
    #     ref_track_ids=np.arange(len(self.tracklet_db))[ref_mask]
    #     return ref_bbox,ref_track_ids


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

from sklearn.metrics.pairwise import cosine_similarity
def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    # if len(stracksa)>0 and len(stracksb)>0:
    #     # fix a derection bug
    #     pcosdist=cosine_similarity(
    #         [track.mean[4:6] for track in stracksa],
    #         [track.mean[4:6] for track in stracksb])
    #     pdist=(pdist+pcosdist)/2
    
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if stracksa[p].mean is not None and stracksb[q].mean is not None:
            x,y=stracksa[p].mean[4:6],stracksa[p].mean[4:6]
            cosine_dist=1-np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-06)
            if cosine_dist>0.15:
                continue
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


