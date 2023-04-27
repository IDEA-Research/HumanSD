import os
import numpy as np
from torch.utils.data import Dataset
import json
import albumentations as A
import cv2
from PIL import Image
import seaborn as sns

# the HumanSD dataset
class HumanSDPoseBase(Dataset):
    def __init__(self,
                 map_file,
                 base_path,
                 image_size,
                 max_person_num,
                 keypoint_num,
                 keypoint_dim,
                 skeleton_width,
                 keypoint_thresh,
                 pose_skeleton
                 ):
        self.base_path = base_path
        with open(map_file, "r",encoding='utf-8') as f:
            self.map_json = json.load(f)
        self._length = len(self.map_json)
        
        data_list=list(self.map_json.keys())
        self.index_to_data = dict((k, data_list[k]) for k in range(self._length))

        self.image_size = image_size
        self.max_person_num=max_person_num
        self.keypoint_num=keypoint_num
        self.keypoint_dim=keypoint_dim
        
        keypoint_index=[ keypoint_i for keypoint_i in range(self.keypoint_num)]*self.max_person_num
        person_index=[ keypoint_i//self.keypoint_num for keypoint_i in range(self.keypoint_num*self.max_person_num)]
        self.person_keypoint_index=[[person_index[ii],keypoint_index[ii]] for ii in range(self.keypoint_num*self.max_person_num)]
        
        self.skeleton_width=skeleton_width
        self.keypoint_thresh=keypoint_thresh
        self.pose_skeleton=pose_skeleton
        self.color=sns.color_palette("hls", len(self.pose_skeleton))
        
        self.transform = A.Compose([
                A.HorizontalFlip(p=0.2), 
                A.VerticalFlip(p=0.1),
                A.Rotate(p=0.1),
                A.SmallestMaxSize(max_size=self.image_size, interpolation=cv2.INTER_AREA),
                A.RandomCrop(width=self.image_size, height=self.image_size),
                A.OneOf([
                    A.HueSaturationValue(p=0.5), 
                    A.RGBShift(p=0.5)
                ], p=0.4),  
                A.RandomBrightnessContrast(p=0.2),
            ], keypoint_params=A.KeypointParams(format='xy', 
                                                label_fields=["person_keypoint_index"], 
                                                remove_invisible=True))   
        

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        def plot_kpts(img_draw, kpts, color=self.color, edgs=self.pose_skeleton,width=self.skeleton_width):     
            for idx, kpta, kptb in edgs:
                if int(kpts[kpta,0])!=-1 and \
                    int(kpts[kpta,1])!=-1 and \
                    int(kpts[kptb,0])!=-1 and \
                    int(kpts[kptb,1])!=-1 and \
                    kpts[kpta,2]>self.keypoint_thresh and \
                    kpts[kptb,2]>self.keypoint_thresh :
                    line_color = tuple([int(255*color_i) for color_i in color[idx]])
                    
                    cv2.line(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), (int(kpts[kptb,0]),int(kpts[kptb,1])), line_color,width)
                    
                    cv2.circle(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), width//2, line_color, -1)
                    cv2.circle(img_draw, (int(kpts[kptb,0]),int(kpts[kptb,1])), width//2, line_color, -1)


        example = self.map_json[self.index_to_data[i]]
        image = Image.open(os.path.join(self.base_path,example["img_path"]))
        if not image.mode == "RGB":
            image = image.convert("RGB")
        
        # default to score-sde preprocessing
        image = np.array(image).astype(np.uint8)
        
        # pose
        detected_results = np.load(os.path.join(self.base_path,example["pose_path"]),allow_pickle=True)["arr_0"]
        
        pose_list=[[0,0]  for ii in range(self.max_person_num*self.keypoint_num)]
        
        for detected_pose_i in range(min(len(detected_results),self.max_person_num)):
            for keypoint_i in range(self.keypoint_num):
                keypoints=detected_results[detected_pose_i]["keypoints"]
                if keypoints[keypoint_i,1]<int(image.shape[0])\
                    and keypoints[keypoint_i,0]<int(image.shape[1])\
                    and keypoints[keypoint_i,1]>=0\
                    and keypoints[keypoint_i,0]>=0:
                    pose_list[detected_pose_i*self.keypoint_num+keypoint_i][0]=keypoints[keypoint_i,0]
                    pose_list[detected_pose_i*self.keypoint_num+keypoint_i][1]=keypoints[keypoint_i,1]
        
        
        # augmentation
        transformed = self.transform(image=image, keypoints=pose_list, person_keypoint_index=self.person_keypoint_index)
        transformed_person_keypoint_index = transformed['person_keypoint_index']
        transformed_keypoints = transformed['keypoints']
        transformed_image = transformed['image'] 
        
        scaled_pose=np.array([-1,-1,0])*np.ones((self.max_person_num,self.keypoint_num,self.keypoint_dim))

        for detected_pose_i in range(min(len(detected_results),self.max_person_num)):
            for keypoint_i in range(self.keypoint_num):
                if [detected_pose_i,keypoint_i] in transformed_person_keypoint_index:
                    pos=transformed_person_keypoint_index.index([detected_pose_i,keypoint_i])
                    if detected_results[detected_pose_i]["keypoints"][keypoint_i,2]>self.keypoint_thresh:
                        scaled_pose[detected_pose_i,keypoint_i,0]=transformed_keypoints[pos][0]
                        scaled_pose[detected_pose_i,keypoint_i,1]=transformed_keypoints[pos][1]
                        scaled_pose[detected_pose_i,keypoint_i,2]=detected_results[detected_pose_i]["keypoints"][keypoint_i,2]
        
        scaled_image=transformed_image

        # normalize to [-1,1]
        return_example={}
        return_example["jpg"] = (scaled_image / 255 *2 - 1.0).astype(np.float32)
        
        return_example["pose"] = scaled_pose
        
        return_example["txt"] = example["prompt"]
        
        pose_img = np.zeros((self.image_size,self.image_size,3))
        for person_i in range(self.max_person_num):
            if np.sum(scaled_pose[person_i,:,:])>0:
                try:
                    plot_kpts(pose_img, scaled_pose[person_i,:,:],self.color,self.pose_skeleton,self.skeleton_width)
                except:
                    print("Can not draw poses ... Skipping.")
        
        return_example["pose_img"]= (pose_img / 255 *2 - 1.0).astype(np.float32)
        
        return return_example


class HumanSDPoseTrain(HumanSDPoseBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class HumanSDPoseValidation(HumanSDPoseBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

