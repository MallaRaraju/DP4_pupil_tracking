'''
Created on 15-May-2023

@author: raraj
'''
import cv2
import json
import albumentations as A
import os

class KP_augmentation():
    def __init__(self):
        self.augmentor = A.Compose([A.RandomBrightness(p=0.5),
                    A.RandomContrast(p=0.5),
                    A.RandomGamma(p=0.2),
                    A.HorizontalFlip(p=0.3),
                    A.VerticalFlip(p=0.3),
                    A.RandomRotate90(p=0.3),
                    A.augmentations.geometric.transforms.Affine(p=0.3,scale=[0.25,0.75]),
                    A.augmentations.geometric.transforms.Affine(p=0.3,scale=[1.25,1.75]),
                    A.augmentations.geometric.transforms.Affine(p=0.3,rotate=[-15,15]) ],
                   keypoint_params=A.KeypointParams(format='xy', label_fields=['point_labels','image_labels']))
    
    def aug_pl(self, label_dir, image_dir):
        labels = os.listdir(label_dir)
        for label in labels:
            with open(os.path.join(label_dir,label)) as f:
                annot = json.load(f)
                stuff = annot['shapes']
                img_path = os.path.join(image_dir,f'{label.split(".")[0]}.jpg')
                img = cv2.imread(img_path)
            if len(stuff)>2:
                coords = [tuple(stuff[0]['points'][0]),tuple(stuff[1]['points'][0])]
                classes = [stuff[0]['label'], stuff[1]['label']]
                img_type = [stuff[2]['label'],stuff[2]['label']]
            else:
                coords = [tuple(stuff[0]['points'][0])]
                classes = [stuff[0]['label']]
                img_type = [stuff[1]['label']]
            for i in range(60):
                annotation = {}
                new_aug = augmentor(image=img,keypoints=coords,point_labels=classes, image_labels = img_type)
                if not len(new_aug['keypoints'])==0:
                    annotation['image']=f'{img_path.split(".")[0]}_{i}.jpg'
                    annotation['points']=new_aug['keypoints']
                    annotation['point_labels']=new_aug['point_labels']
                    annotation['image_label']=new_aug['image_labels'][0]
                    cv2.imwrite(f'{img_path.split(".")[0]}_augmented_{i}.jpg',new_aug['image'])
                    with open(os.path.join('/',root_dir,'data','label',f'{label.split(".")[0]}_augmentged_{i}.json'), 'w') as o:
                        json.dump(annotation,o)
