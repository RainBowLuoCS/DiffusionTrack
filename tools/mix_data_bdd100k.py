import json
import os
import numpy as np

"""
cd datasets
mkdir -p mix_det/annotations
cp mot/annotations/val_half.json mix_det/annotations/val_half.json
cp mot/annotations/test.json mix_det/annotations/test.json
cd mix_det
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
ln -s ../Cityscapes cp_train
ln -s ../ETHZ ethz_train
cd ..
"""

bdd100ktrain_json = json.load(open('datasets/bdd100k/annotations/mix_train_val.json','r'))
# need_index=np.random.choice(range(len(bdd100ktrain_json['images'])),len(bdd100ktrain_json['images'])//3,replace=False)
# need_img_ids={}
img_list = list()
for img in bdd100ktrain_json['images']:
    img['is_video']=1
    img_list.append(img)
    # need_img_ids[bdd100ktrain_json['images'][img_idx]['id']]=1

ann_list = list()
for ann in bdd100ktrain_json['annotations']:
    # if ann['image_id'] in need_img_ids:
    ann_list.append(ann)

video_list = bdd100ktrain_json['videos']
category_list = bdd100ktrain_json['categories']


print('bdd100ktrain')

max_img = len(img_list)
max_ann = len(ann_list)
max_video = len(video_list)

bdd100kval_json = json.load(open('datasets/bdd100k/annotations/val.json','r'))
for img in bdd100kval_json['images']:
    img['prev_image_id'] = img['prev_image_id'] + max_img
    img['next_image_id'] = img['next_image_id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id']+= max_video
    img['is_video']=1
    img_list.append(img)
    
for ann in bdd100kval_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

for vid in bdd100kval_json['videos']:
    vid['id']+=max_video
    video_list.append(vid)

print('bdd100ktest')

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/bdd100k/annotations/mix_train_val.json','w'))
