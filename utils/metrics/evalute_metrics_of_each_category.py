import json
import os
import pandas as pd
import torchvision.transforms as TF
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.utilities.data import dim_zero_cat
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gen",default="z_logs_new/metrics/union_pose_loss_back/54400_metrics-union-embedd-pose_loss_thresh01_back_to_embed_w01-data_thresh002")
args = parser.parse_args()

humanart_validation_set="sd_data/datasets/HumanArt/mapping_validation.json"

gt_dir="sd_data/datasets/HumanArt"
generate_dir=args.gen
save_csv=os.path.join(generate_dir,"style_quality.csv")
with open(humanart_validation_set,"r") as f:
    humanart_validation=json.load(f)
    

humanart_scene_split={}

for k,image in humanart_validation.items():
    present_style=image["img_path"].split(os.sep)[-2]
    
    if present_style in humanart_scene_split.keys():
        humanart_scene_split[present_style].append(image)
    else:
        humanart_scene_split[present_style]=[image]

df = pd.DataFrame(columns=['style','fid','kid_1','kid_2','is_1','is_2'])


for style,images in humanart_scene_split.items():
    print(f"present style is: {style}")
    
    device="cuda"
    
    fid_image_transforms=TF.Compose([
            TF.Resize(299),
            TF.CenterCrop(299),
            TF.ToTensor(),
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    fid_image_transforms_gen=TF.Compose([
            TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    print("initialize contrast dataset")
    
    dataset_imgs=[]
    
    for image_i in tqdm(images):
        present_image_path = os.path.join(gt_dir,image_i["img_path"])
        img = Image.open(present_image_path).convert('RGB')
        dataset_imgs.append(img)
        
    dataset_imgs_tensor=[]
    
    for image_i in tqdm(dataset_imgs):
        dataset_imgs_tensor.append(fid_image_transforms(image_i).unsqueeze(0))
        
    dataset_imgs_tensor=torch.concat(dataset_imgs_tensor).to(device)
    
    generated_imgs_tensor=[]
    for image_i in tqdm(images):
        generate_image_path = os.path.join(generate_dir,image_i["img_path"])
        img = Image.open(generate_image_path).convert('RGB')
        generated_imgs_tensor.append(np.array(img.resize((299,299))))
    
    generated_imgs_tensor=torch.tensor(generated_imgs_tensor).permute(0,3,1,2)/255
    
    gen_dataset_imgs_tensor=fid_image_transforms_gen(generated_imgs_tensor)
    
    
    fid_model_feature=64
    fid_model=FrechetInceptionDistance(feature=fid_model_feature,normalize=True).to(device)
    fid_model.update(dataset_imgs_tensor,real=True)
    fid_model.update(gen_dataset_imgs_tensor, real=False)
    
    fid_results=fid_model.compute().item()
    
    # IS
    inception_model = InceptionScore(normalize=True).to(device)
    inception_model.update(gen_dataset_imgs_tensor)
    is_results=inception_model.compute()
    is_results_1=is_results[0].item()
    is_results_2=is_results[1].item()
    
    # KID
    kid_model = KernelInceptionDistance(kid_subset_size=250,normalize=True).to(device)
    kid_model.update(dataset_imgs_tensor, real=True)
    kid_model.update(gen_dataset_imgs_tensor, real=False)
    kid_results=kid_model.compute()
    kid_results_1=kid_results[0].item()
    kid_results_2=kid_results[1].item()
        
    df=df.append([{
        'style':style,
        'fid':fid_results,
        'kid_1':kid_results_1,
        'kid_2':kid_results_2,
        'is_1':is_results_1,
        'is_2':is_results_2
    }], ignore_index = True)
    
    df.to_csv(save_csv)
    print("finish")
    print("============================================")