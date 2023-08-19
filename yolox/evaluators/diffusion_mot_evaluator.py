from collections import defaultdict
from loguru import logger
from tqdm import tqdm

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)
from yolox.tracker.diffusion_tracker import DiffusionTracker
from yolox.models import  YOLOXHead

import contextlib
import io
import os
import itertools
import json
import tempfile
import time
import numpy as np

def write_results(filename, results):
    save_format = '{frame},{id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{s:.2f},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class DiffusionMOTEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self, args, dataloader, img_size, confthre, nmsthre3d, detthre,nmsthre2d,interval, num_classes):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre3d = nmsthre3d
        self.detthre=detthre
        self.nmsthre2d=nmsthre2d
        self.num_classes = num_classes
        self.association_interval=interval
        self.args = args

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        result_folder=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        results = []
        seq_ids=[]
        seq_info_imgs=[]
        seq_frame_ids=[]
        video_names = defaultdict()
        ori_detthre=self.detthre
        ori_confthre=self.confthre
        progress_bar = tqdm if is_main_process() else iter

        track_time = 0
        n_samples = len(self.dataloader) - 1

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
            
        tracker = DiffusionTracker(model,tensor_type)
        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                # init tracker
                frame_id = info_imgs[2].item()
                video_id = info_imgs[3].item()
                img_file_name = info_imgs[4]
                video_name = img_file_name[0].split('/')[0]

                if video_name not in video_names:
                    video_names[video_id] = video_name
                
                self.detthre=ori_detthre
                self.confthre=ori_confthre
                # if video_name == 'MOT17-02-FRCNN' or video_name == 'MOT17-01-FRCNN':
                # #self.association_interval = 1
                #     self.confthre = 0.6
                # elif video_name == 'MOT17-04-FRCNN' or video_name=='MOT17-03-FRCNN':
                #     self.detthre = 0.5
                #     #self.association_interval = 1
                #     self.confthre = 0.4
                # elif video_name == 'MOT17-05-FRCNN' or video_name == 'MOT17-06-FRCNN':
                #     #self.association_interval = 1
                #     self.confthre = 0.4
                # elif video_name == 'MOT17-09-FRCNN' or video_name == 'MOT17-07-FRCNN':
                #     self.confthre = 0.3
                #     self.detthre = 0.5
                #     self.nmsthre3d = 0.6
                # elif video_name == 'MOT17-10-FRCNN' or video_name == 'MOT17-8-FRCNN':
                #     self.confthre = 0.4
                #     self.nmsthre3d = 0.5
                # elif video_name == 'MOT17-11-FRCNN' or video_name == 'MOT17-12-FRCNN':
                #     self.confthre = 0.5
                # elif video_name == 'MOT17-13-FRCNN' or video_name == 'MOT17-14-FRCNN':
                #     self.confthre = 0.5
                #     self.detthre = 0.5
                # if video_name =="MOT20-06" or video_name=="MOT20-08":
                #     self.detthre=0.3

                # if video_name!="MOT20-01" and video_name!="MOT20-02":
                #     continue

                if frame_id == 1:
                    if len(seq_ids) != 0:
                        outputs=tracker.get_results()
                        track_time+=tracker.total_time
                        result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id - 1]))
                        for output,info_img,id,cur_frame_id in zip(outputs,seq_info_imgs,seq_ids,seq_frame_ids):
                            output_results,scale = self.convert_to_coco_format(output, info_img, id)
                            data_list.extend(output_results)

                            # run tracking
                            online_tlwhs = []
                            online_ids = []
                            online_scores = []
                            for tid,obj in zip(*output):
                                xyxy = obj[:4]/scale
                                tlwh = [xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]]
                                vertical = tlwh[2] / tlwh[3] > 1.6
                                if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                                    online_tlwhs.append(tlwh)
                                    online_ids.append(tid)
                                    online_scores.append(obj[4])
                                # save results
                            results.append((cur_frame_id, online_tlwhs, online_ids, online_scores))
                        write_results(result_filename, results)
                        results = []
                        seq_ids=[]
                        seq_info_imgs=[]
                        seq_frame_ids=[]
                    tracker = DiffusionTracker(model,tensor_type,self.confthre,self.detthre,self.nmsthre3d,self.nmsthre2d,self.association_interval)

                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference

                tracker.update(imgs)
            

            seq_ids.append(ids)
            seq_info_imgs.append(info_imgs)
            seq_frame_ids.append(frame_id)
            
            if cur_iter == len(self.dataloader) - 1:
                result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
                outputs=tracker.get_results()
                track_time+=tracker.total_time
                for output,info_img,id,cur_frame_id in zip(outputs,seq_info_imgs,seq_ids,seq_frame_ids):
                        output_results,scale = self.convert_to_coco_format(output, info_img, id)
                        data_list.extend(output_results)
                        # run tracking
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        for tid,obj in zip(*output):
                            xyxy = obj[:4]/scale
                            tlwh = [xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]]
                            vertical = tlwh[2] / tlwh[3] > 1.6
                            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(obj[4])
                            # save results
                        results.append((cur_frame_id, online_tlwhs, online_ids, online_scores))
                write_results(result_filename, results)

        print("diffusion track fps : {}".format(n_samples*2/track_time))
        
        statistics = torch.cuda.FloatTensor([0, track_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, output, info_imgs, ids):
        data_list = []
        scale = min(
                self.img_size[0] / float(info_imgs[0]), self.img_size[1] / float(info_imgs[1])
            )
        bboxes = []
        clses = []
        scores = []

        if len(output[1])>0:
            for t in output[1]:
                bboxes.append(t[:4])
                clses.append(0)
                scores.append(t[4])
            bboxes=np.array(bboxes)
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)
            
        for ind in range(len(bboxes)):
            label = self.dataloader.dataset.class_ids[int(clses[ind])]
            pred_data = {
                "image_id": int(ids[0]),
                "category_id": label,
                "bbox": bboxes[ind].tolist(),
                "score": float(scores[ind]),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)
        return data_list,scale

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        track_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_track_time = 1000 * track_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "track", "inference"],
                    [a_infer_time, a_track_time, (a_infer_time + a_track_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            '''
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools import cocoeval as COCOeval
                logger.warning("Use standard COCOeval.")
            '''
            #from pycocotools.cocoeval import COCOeval
            from yolox.layers import COCOeval_opt as COCOeval
            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
