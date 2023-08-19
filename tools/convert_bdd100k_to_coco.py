import cv2
import os
import json
import tqdm
import numpy as np

labels_path = 'datasets/bdd100k/labels'
img_path = 'datasets/bdd100k/images'
# mot_labels_path  = '/data/yourname/BDD100K-MOT/GT'

out_path = 'datasets/bdd100k/annotations/'

split = ['train']
categories = [
    {"id": 1, "name": "pedestrian"},
    {"id": 2, "name": "rider"},
    {"id": 3, "name": "car"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bus"},
    {"id": 6, "name": "train"},
    {"id": 7, "name": "motorcycle"},
    {"id": 8, "name": "bicycle"},
    # {"id": 9, "name": "traffic light"},
    # {"id": 10, "name": "traffic sign"},
]

# "traffic light":9, "traffic sign":10
cat = {"pedestrian":1, "rider":2, "car":3, "truck":4, "bus":5, "train":6, "motorcycle":7, "bicycle":8,}
# 1: pedestrian
# 2: rider
# 3: car
# 4: truck
# 5: bus
# 6: train
# 7: motorcycle
# 8: bicycle  
# 9: traffic light --- Don't need tracking
# 10: traffic sign  ---   Don't need tracking
# For MOT and MOTS, only the first 8 classes are used and evaluated

def read_tid_num_per_video(video_ann_dir):
    anns = np.loadtxt(video_ann_dir, dtype=np.float32, delimiter=',')
    max_tid = max(anns[:, 1])
    return int(max_tid)
    

for s in split:
    img_id = 1; ann_id = 1; video_cnt = 0; 
    tid_cnt = 0 
    images = []; annotations=[]; videos = []
    all_video=[d for d in os.listdir(os.path.join(labels_path, s)) if '.json' in d]
    need_index=np.random.choice(range(len(all_video)),len(all_video)//3,replace=False)
    video_labels_list = [all_video[i] for i in need_index]
    
    for v_label in tqdm.tqdm(video_labels_list):
        video_cnt += 1
        video = {'id': video_cnt, 'file_name':v_label[:-5]}
        videos.append(video)
        
        v_lab_path = os.path.join(os.path.join(labels_path, s, v_label))
        with open(v_lab_path, 'r') as f:
            annos=json.load(f)# anns per video
        num_frames  = len(annos)# the number of frames per video
        sign_cnt = 0
        for ann in annos:# ann --- 每一帧的标注信息，这里放过了空白帧
            
            img_name = os.path.join(img_path, s, ann['videoName'], ann['name'])
            img=cv2.imread(img_name)
            h,w,_ = img.shape
            
            img_info = {
            'file_name':img_name,
            'width':w,
            'height':h,
            'id': img_id,
            'frame_id': ann['frameIndex'] + 1,# 严格按照 数据集 标记的帧indx 来进行排序，这将有利于 判断 相邻帧 之间的关系
            'prev_image_id': -1 if ann['frameIndex'] == 0 else img_id - 1,
            'next_image_id': -1 if ann['frameIndex'] == num_frames-1 else img_id + 1,
            'video_id': video_cnt
            }# 所有的图像信息images中 ，这里也会添加空白标注帧的图像信息
            images.append(img_info)
            
            for j, lab in enumerate(ann['labels']):
                #  lab---每一个实例的标注信息  如果遇到空白标注帧--ann['labels']为空 则循环不执行 如果帧为非空 则继续执行此循环
                if lab['category'] in cat:# 为了避免 'other vehicle' 类
                    pass
                else:
                    continue
                    
                track_id = lab['id']
                     
                if sign_cnt == 0 and j==0:
                    firstid = track_id
                    sign_cnt = 1      
                     
                tid_curr = int(track_id) - int(firstid) + 1
                tid_cnt+=1
                is_crowd = lab['attributes']['crowd']
                x1, y1, x2, y2=lab['box2d']['x1'], lab['box2d']['y1'], lab['box2d']['x2'], lab['box2d']['y2']
                
                annotation = {
                    'image_id': img_id,
                    'conf': 1,
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'category_id': cat[lab['category']],
                    'id': ann_id,
                    'iscrowd':  1 if is_crowd else 0,
                    'track_id': tid_curr + tid_cnt,
                    'segmentation': [],
                    'area': (x2-x1)*(y2-y1),
                    'box_id':int(track_id)   
                }
                annotations.append(annotation)
                ann_id += 1
                    
            img_id += 1
            
        # tid_cnt += read_tid_num_per_video(os.path.join(mot_labels_path, s, v_label[:-5]+'.txt'))
            
    dataset_dict = {}
    dataset_dict["images"] = images
    dataset_dict["annotations"] = annotations
    dataset_dict["categories"] = categories
    dataset_dict["videos"] = videos
    
    json_str = json.dumps(dataset_dict)
    print(f' The number of detection objects is {ann_id - 1}, The number of detection imgs is {img_id -1} .')
    with open(out_path+f'{s}.json', 'w') as json_file:
        json_file.write(json_str)