'''
Created on 15-May-2023

@author: raraj
'''
import albumentations as A
import os
import json
import cv2

class KP_augmentation():
    def __init__(self):
        self.augmentor = A.Compose([A.RandomBrightness(p=0.5),
                        A.RandomContrast(p=0.5),
                        A.RandomGamma(p=0.2),
                        A.RandomFog(p=0.1),
                        A.RandomRain(p=0.3),
                        A.RandomToneCurve(p=0.3),
                        A.HorizontalFlip(p=0.5),
                        A.VerticalFlip(p=0.5),
                        A.RandomRotate90(p=0.3)],
                       keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']),)
    
    def aug_pl(self, label_dir):
        labels = os.listdir(label_dir)
        root_dir = label_dir.split("/")[0]
        for label in labels:
            with open(os.path.join(label_dir,label)) as f:
                annot = json.load(f)
                stuff = annot['shapes']
                img_path = os.path.join(root_dir,'data', annot['imagePath'][3:])
                img = cv2.imread(img_path)
                if len(stuff)>1:
                    coords = [tuple(stuff[0]['points'][0]),tuple(stuff[1]['points'][0])]
                    classes = [stuff[0]['label'], stuff[1]['label']]
                else:
                    coords = [tuple(stuff[0]['points'][0])]
                    classes = [stuff[0]['label']]
                for i in range(60):
                    annotation = {}
                    new_aug = self.augmentor(image=img,keypoints=coords,class_labels=classes)
                    annotation['image']=f'{img_path.split(".")[0]}_{i}.jpg'
                    annotation['points']=new_aug['keypoints']
                    annotation['labels']=new_aug['class_labels']
                    cv2.imwrite(f'{img_path.split(".")[0]}_{i}.jpg',new_aug['image'])
                    with open(os.path.join(root_dir,'data','label',f'{img_path.split(".")[0]}_{i}.json'), 'w') as o:
                        json.dump(annotation,o)
