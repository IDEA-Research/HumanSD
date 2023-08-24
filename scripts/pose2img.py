import torch
from pytorch_lightning import seed_everything

import gradio as gr

import argparse
import sys
import numpy as np
import math
from einops import repeat, rearrange
import os
import time 

import cv2
import seaborn as sns
from PIL import Image


# HumanSD
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler as humansd_DDIMSampler
from imwatermark import WatermarkEncoder
from scripts.txt2img import put_watermark

# pose detector
from mmpose.apis import inference_bottom_up_pose_model, init_pose_model


DEVICE="cuda" if torch.cuda.is_available() else "cpu"
IMAGE_RESOLUTION=512

def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img

def draw_humansd_skeleton(image, present_pose,mmpose_detection_thresh):
    humansd_skeleton=[
              [0,0,1],
              [1,0,2],
              [2,1,3],
              [3,2,4],
              [4,3,5],
              [5,4,6],
              [6,5,7],
              [7,6,8],
              [8,7,9],
              [9,8,10],
              [10,5,11],
              [11,6,12],
              [12,11,13],
              [13,12,14],
              [14,13,15],
              [15,14,16],
          ]
    humansd_skeleton_width=10
    humansd_color=sns.color_palette("hls", len(humansd_skeleton)) 
    
    def plot_kpts(img_draw, kpts, color, edgs,width):     
            for idx, kpta, kptb in edgs:
                if kpts[kpta,2]>mmpose_detection_thresh and \
                    kpts[kptb,2]>mmpose_detection_thresh :
                    line_color = tuple([int(255*color_i) for color_i in color[idx]])
                    
                    cv2.line(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), (int(kpts[kptb,0]),int(kpts[kptb,1])), line_color,width)
                    cv2.circle(img_draw, (int(kpts[kpta,0]),int(kpts[kpta,1])), width//2, line_color, -1)
                    cv2.circle(img_draw, (int(kpts[kptb,0]),int(kpts[kptb,1])), width//2, line_color, -1)
    
     
    pose_image = np.zeros_like(image)
    for person_i in range(len(present_pose)):
        if np.sum(present_pose[person_i]["keypoints"])>0:
            plot_kpts(pose_image, present_pose[person_i]["keypoints"],humansd_color,humansd_skeleton,humansd_skeleton_width)
    
    return pose_image


def draw_controlnet_skeleton(image, pose,mmpose_detection_thresh):
    H, W, C = image.shape
    canvas = np.zeros((H, W, C))
    
    for pose_i in range(len(pose)):
        present_pose=pose[pose_i]["keypoints"]
        candidate=[
                [present_pose[0,0],present_pose[0,1],present_pose[0,2],0],
                [(present_pose[6,0]+present_pose[5,0])/2,(present_pose[6,1]+present_pose[5,1])/2,(present_pose[6,2]+present_pose[5,2])/2,1] if present_pose[6,2]>mmpose_detection_thresh and present_pose[5,2]>mmpose_detection_thresh else [-1,-1,0,1],
                [present_pose[6,0],present_pose[6,1],present_pose[6,2],2],
                [present_pose[8,0],present_pose[8,1],present_pose[8,2],3],
                [present_pose[10,0],present_pose[10,1],present_pose[10,2],4],
                [present_pose[5,0],present_pose[5,1],present_pose[5,2],5],
                [present_pose[7,0],present_pose[7,1],present_pose[7,2],6],
                [present_pose[9,0],present_pose[9,1],present_pose[9,2],7],
                [present_pose[12,0],present_pose[12,1],present_pose[12,2],8],
                [present_pose[14,0],present_pose[14,1],present_pose[14,2],9],
                [present_pose[16,0],present_pose[16,1],present_pose[16,2],10],
                [present_pose[11,0],present_pose[11,1],present_pose[11,2],11],
                [present_pose[13,0],present_pose[13,1],present_pose[13,2],12],
                [present_pose[15,0],present_pose[15,1],present_pose[15,2],13],
                [present_pose[2,0],present_pose[2,1],present_pose[2,2],14],
                [present_pose[1,0],present_pose[1,1],present_pose[1,2],15],
                [present_pose[4,0],present_pose[4,1],present_pose[4,2],16],
                [present_pose[3,0],present_pose[3,1],present_pose[3,2],17],
                ]
        stickwidth = 4
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        for i in range(17):
            if candidate[limbSeq[i][0]-1][2]>mmpose_detection_thresh and candidate[limbSeq[i][1]-1][2]>mmpose_detection_thresh:
                Y=[candidate[limbSeq[i][1]-1][0],candidate[limbSeq[i][0]-1][0]]
                X=[candidate[limbSeq[i][1]-1][1],candidate[limbSeq[i][0]-1][1]]

                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
                cur_canvas = canvas.copy()
                cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
                canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

        for i in range(18):
            if candidate[i][2]>mmpose_detection_thresh:
                x, y = candidate[i][0:2]
                cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)

    return canvas

def make_batch_sd(
        image,
        pose_image,
        txt,
        device,
        num_samples=1,
):
    batch={
        "jpg":(torch.from_numpy(image).to(dtype=torch.float32) / 255 *2 - 1.0),
        "pose_img": (torch.from_numpy(pose_image).to(dtype=torch.float32) / 255 *2 - 1.0),
        "txt": num_samples * [txt],
    }
    
    batch["pose_img"] = rearrange(batch["pose_img"], 'h w c -> 1 c h w')
    batch["pose_img"] = repeat(batch["pose_img"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    return batch


def paint_humansd(humansd_sampler, image, pose_image, prompt, t_enc, seed, scale, device, num_samples=1, callback=None,
          do_full_sample=False,negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"):
    model = humansd_sampler.model
    seed_everything(seed)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "HumanSD"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    with torch.no_grad():
        batch = make_batch_sd(
            image,pose_image, txt=prompt, device=device, num_samples=num_samples)
        z = model.get_first_stage_encoding(model.encode_first_stage(
            batch[model.first_stage_key]))  # move to latent space
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck]
            if len(cc.shape) == 3:
                cc = cc[..., None]
            cc = cc.to(memory_format=torch.contiguous_format).float()
            cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
            c_cat.append(cc)
            
        c_cat = torch.cat(c_cat, dim=1)
        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, negative_prompt)
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        if not do_full_sample:
            # encode (scaled latent)
            z_enc = humansd_sampler.stochastic_encode(
                z, torch.tensor([t_enc] * num_samples).to(model.device))
        else:
            z_enc = torch.randn_like(z)
        # decode it
        samples = humansd_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full, callback=callback)
        x_samples_ddim = model.decode_first_stage(samples)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [pose_image.astype(np.uint8)] + [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]




def paint_controlnet(controlnet_sampler, pose_image,prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta):

    with torch.no_grad():
        H, W, C = pose_image.shape

        control = torch.from_numpy(pose_image.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = rearrange(control, 'b h w c -> b c h w').clone()

        seed_everything(seed)

        cond = {"c_concat": [control], "c_crossattn": [controlnet_model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [controlnet_model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        strength=1
        controlnet_model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples, intermediates = controlnet_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)


        x_samples = controlnet_model.decode_first_stage(samples)
        x_samples = (rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [Image.fromarray(x_samples[i].astype(np.uint8)) for i in range(num_samples)]
    return [pose_image.astype(np.uint8)] + results

def paint_t2i(t2i_pose_image,num_samples,prompt,added_prompt,negative_prompt,ddim_steps,scale,seed):
    from comparison_models.T2IAdapter.ldm.inference_base import diffusion_inference as t2i_diffusion_inference
    from comparison_models.T2IAdapter.ldm.modules.extra_condition.api import get_adapter_feature as get_t2i_adapter_feature

    H, W, C = t2i_pose_image.shape

    t2i_control = torch.from_numpy(t2i_pose_image.copy()).float().cuda() / 255.0
    t2i_control = torch.stack([t2i_control for _ in range(1)], dim=0)
    t2i_control = rearrange(t2i_control, 'b h w c -> b c h w').clone()
    
    t2i_opt.prompt=prompt + ", "+added_prompt
    t2i_opt.neg_prompt=negative_prompt
    t2i_opt.steps=ddim_steps
    t2i_opt.max_resolution=IMAGE_RESOLUTION*IMAGE_RESOLUTION
    t2i_opt.scale=scale
    t2i_opt.seed=seed
    t2i_opt.n_samples=num_samples
    t2i_opt.style_cond_tau=1.0
    t2i_opt.cond_tau=1.0
    t2i_opt.H=H
    t2i_opt.W=W
    
    result= [t2i_pose_image.astype(np.uint8)]
    
    for idx in range(t2i_opt.n_samples):
    
        adapter_features, append_to_context = get_t2i_adapter_feature(t2i_control, t2i_adapter)
        
        x_samples = t2i_diffusion_inference(t2i_opt, t2i_sd_model, t2i_sampler, adapter_features, append_to_context)
        x_samples = (rearrange(x_samples, 'b c h w -> b h w c') * 255.).cpu().numpy().clip(0, 255).astype(np.uint8)

        x_samples = [Image.fromarray(x_samples[i].astype(np.uint8)) for i in range(1)]
        result= result + x_samples
    
    return result

def predict(args):
    image = np.zeros((args.image_H, args.image_W, 3))
    image = resize_image(image,IMAGE_RESOLUTION)  # resize to integer multiple of 32
    
    humansd_result=[]
    controlnet_result=[]
    t2i_result=[]
    
    mmpose_results=np.load(args.pose_file,allow_pickle=True)["arr_0"]
    mmpose_filtered_results=[]
    for mmpose_result in mmpose_results:
        if mmpose_result["score"]>args.detection_thresh:
            mmpose_filtered_results.append(mmpose_result)
            
    humansd_pose_image=draw_humansd_skeleton(image,mmpose_filtered_results,args.detection_thresh)
    if args.controlnet:
        controlnet_pose_image=draw_controlnet_skeleton(image,mmpose_filtered_results,args.detection_thresh)
    if args.t2i:
        t2i_pose_image=draw_controlnet_skeleton(image,mmpose_filtered_results,args.detection_thresh)
    
    # humansd
    humansd_sampler.make_schedule(args.ddim_steps, ddim_eta=args.eta, verbose=True)
    do_full_sample = args.strength == 1.
           
    t_enc = min(int(args.strength * args.ddim_steps), args.ddim_steps-1)
    humansd_result = paint_humansd(
        humansd_sampler=humansd_sampler,
        image=image,
        pose_image=humansd_pose_image,
        prompt=args.prompt + ", "+args.added_prompt,
        t_enc=t_enc,
        seed=args.seed,
        scale=args.scale,
        num_samples=args.num_samples,
        callback=None,
        do_full_sample=do_full_sample,
        device=DEVICE,
        negative_prompt=args.negative_prompt
    )
    
    if args.controlnet:
        controlnet_result = paint_controlnet(
            controlnet_sampler=controlnet_sampler, 
            pose_image=controlnet_pose_image,
            prompt=args.prompt, 
            a_prompt=args.added_prompt, 
            n_prompt=args.negative_prompt, 
            num_samples=args.num_samples, 
            ddim_steps=args.ddim_steps, 
            guess_mode=False, 
            strength=args.strength, 
            scale=args.scale, 
            seed=args.seed, 
            eta=args.eta)
        
    if args.t2i:
        t2i_result=paint_t2i(
            t2i_pose_image=t2i_pose_image,
            num_samples=args.num_samples,
            prompt=args.prompt,
            added_prompt=args.added_prompt,
            negative_prompt=args.negative_prompt,
            ddim_steps=args.ddim_steps,
            scale=args.scale,
            seed=args.seed)
        
    print(f"Images are save in {args.save_path}")
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        
    
    save_name=time.strftime("%Y-%m-%d-%H_%M_%S.jpg", time.localtime(time.time()))
    if args.controlnet:
        if args.t2i:
            save_image=np.concatenate((np.concatenate(humansd_result,0),np.concatenate(controlnet_result,0),np.concatenate(t2i_result,0)),1)
        else:
            save_image=np.concatenate((np.concatenate(humansd_result,0),np.concatenate(controlnet_result,0)),1)
    elif args.t2i:
            save_image=np.concatenate((np.concatenate(humansd_result,0),np.concatenate(t2i_result,0)),1)
    else:
        save_image=np.concatenate(humansd_result,0)
            
    save_image=save_image[...,[2,1,0]]
    
    cv2.imwrite(os.path.join(args.save_path,save_name),save_image)
    
    return humansd_result,controlnet_result,t2i_result
    
    
    

def main(args):
    if args.controlnet:
        sys.path.append("comparison_models/ControlNet")
        from comparison_models.ControlNet.cldm.model import create_model as create_controlnet_model , load_state_dict as load_controlnet_state_dict
        from comparison_models.ControlNet.cldm.ddim_hacked import DDIMSampler as controlnet_DDIMSampler
        CONTROLNET_MODEL_CONFIG='comparison_models/ControlNet/models/cldm_v15.yaml'
        CONTROLNET_BASE_MODEL_CKPT='humansd_data/checkpoints/control_sd15_openpose.pth'
        CONTROLNET_MODEL_CKPT='humansd_data/checkpoints/v1-5-pruned.ckpt'
        global controlnet_sampler
        global controlnet_model
        controlnet_model = create_controlnet_model(CONTROLNET_MODEL_CONFIG).cpu()
        controlnet_model.load_state_dict(load_controlnet_state_dict(CONTROLNET_BASE_MODEL_CKPT,location=DEVICE), strict=False)
        controlnet_model.load_state_dict(load_controlnet_state_dict(CONTROLNET_MODEL_CKPT,location=DEVICE), strict=False)
        controlnet_model = controlnet_model.to(DEVICE)
        controlnet_sampler = controlnet_DDIMSampler(controlnet_model)
    if args.t2i:
        # t2i
        sys.path.append("comparison_models/T2IAdapter")
        from comparison_models.T2IAdapter.ldm.inference_base import get_adapters as get_t2i_adapters, get_sd_models as get_t2i_sd_models
        from comparison_models.T2IAdapter.ldm.modules.extra_condition.api import ExtraCondition as t2i_ExtraCondition

        global t2i_sd_model
        global t2i_sampler
        global t2i_adapter
        global t2i_opt
        class T2I_OPT():
            def __init__(self) -> None:
                self.which_cond="openpose"
                self.sd_ckpt="humansd_data/checkpoints/v1-5-pruned.ckpt"
                self.adapter_ckpt="humansd_data/checkpoints/t2iadapter_openpose_sd14v1.pth"
                self.config="comparison_models/T2IAdapter/configs/stable-diffusion/sd-v1-inference.yaml"
                self.vae_ckpt=None
                self.device=DEVICE
                self.cond_weight=1.0
                self.sampler='ddim'
                self.C=4
                self.f=8
                
        t2i_opt=T2I_OPT()

        t2i_sd_model, t2i_sampler = get_t2i_sd_models(t2i_opt)
        t2i_adapter = get_t2i_adapters(t2i_opt, getattr(t2i_ExtraCondition, t2i_opt.which_cond)) 
        
    global humansd_sampler
    humansd_config=OmegaConf.load(args.humansd_config)
    humansd_model = instantiate_from_config(humansd_config.model)
    humansd_model.load_state_dict(torch.load(args.humansd_checkpoint)["state_dict"], strict=False)
    humansd_model = humansd_model.to(DEVICE)
    humansd_sampler = humansd_DDIMSampler(humansd_model)
    
    global mmpose_model
    mmpose_model=init_pose_model(args.mmpose_config, args.mmpose_checkpoint, device=DEVICE)
               
    fn=predict(args)
    
    


if __name__=="__main__":
    print("You are running the demo of HumanSD ........")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        help="the prompt"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
        help="the negative prompt"
    )
    parser.add_argument(
        "--added_prompt",
        type=str,
        default="detailed background, best quality, extremely detailed",
        help="the added prompt"
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        help="the pose npz file in MMPose format"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="outputs/visualization_results",
        help="the image save path"
    )
    parser.add_argument(
        "--image_H",
        type=int,
        default=512,
        help="number of samples"
    )
    parser.add_argument(
        "--image_W",
        type=int,
        default=512,
        help="number of samples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="ddim step number"
    )
    parser.add_argument(
        "--detection_thresh",
        type=float,
        default=0.05,
        help="detection threshold"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="guidance scale"
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="pose strength"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta"
    )
    parser.add_argument(
        "--humansd_config",
        type=str,
        default="configs/humansd/humansd-inference.yaml",
        help="the config file of HumanSD"
    )
    parser.add_argument(
        "--humansd_checkpoint",
        type=str,
        default="humansd_data/checkpoints/humansd-v1.ckpt",
        help="the checkpoint of HumanSD"
    )
    parser.add_argument(
        "--mmpose_config",
        type=str,
        default="humansd_data/models/mmpose/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512_udp.py",
        help="the config file of human pose estimator"
    )
    parser.add_argument(
        "--mmpose_checkpoint",
        type=str,
        default="humansd_data/checkpoints/higherhrnet_w48_coco_512x512_udp.pth",
        help="the checkpoint of human pose estimator",
    )
    
    parser.add_argument(
        "--controlnet",
        action='store_true',
        help="generate images from ControlNet",
    )
    parser.add_argument(
        "--t2i",
        action='store_true',
        help="generate images from T2I-Adapter",
    )
    
    args = parser.parse_args()
    main(args)