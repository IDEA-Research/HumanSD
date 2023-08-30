import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from PIL import Image, ImageDraw, ImageFont
import einops
import cv2

def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('DejaVuSans.ttf', size=size)
        nc = int(20 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((30, 30), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model



def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", 
                        type=str, 
                        nargs="?",
                        help="dir to write results to", 
                        default="outputs/eval")
    parser.add_argument("--metrics", 
                        default=["pose","quality","text"])
    parser.add_argument("--ddim_steps", 
                        type=int, 
                        default=50,
                        help="number of ddim sampling steps")
    parser.add_argument( "--plms", 
                        action='store_true',
                        help="use plms sampling")
    parser.add_argument("--dpm_solver", 
                        action='store_true',
                        help="use dpm_solver sampling")
    parser.add_argument("--ddim_eta", 
                        type=float, 
                        default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--H", 
                        type=int, 
                        default=512,
                        help="image height, in pixel space")
    parser.add_argument("--W", 
                        type=int, 
                        default=512,
                        help="image width, in pixel space")
    parser.add_argument("--C", 
                        type=int, 
                        default=4,
                        help="latent channels")
    parser.add_argument( "--f", 
                        type=int, 
                        default=8,
                        help="downsampling factor")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=10,
                        help="how many samples to produce for each given prompt. A.k.a. batch size")
    parser.add_argument("--scale", 
                        type=float, 
                        default=9,
                        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    parser.add_argument("--config", 
                        type=str, 
                        help="path to config which constructs model")
    parser.add_argument("--ckpt", 
                        type=str, 
                        help="path to checkpoint of model")
    parser.add_argument("--seed", 
                        type=int, 
                        default=42,
                        help="the seed (for reproducible sampling)")
    parser.add_argument("--device", 
                        type=str, 
                        default="cuda")
    parser.add_argument("--device_ids", 
                        type=str, 
                        default=[0,1,2,3])
    
    opt = parser.parse_args()
    
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    model = model.to(opt.device)
    
    metrics_calculator=instantiate_from_config(config.metrics)
    
    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
        
    sampler=torch.nn.DataParallel(sampler,device_ids=opt.device_ids)
        
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.batch_size
    
    start_code = None
    
    dataset = instantiate_from_config(config.data.params.validation)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_metrics_results={metric:{} for metric in opt.metrics}
    
    with torch.no_grad():
        for batch_idx, (data) in enumerate(test_loader):
            with model.ema_scope():
                input_data=model.get_input(data,0)
                
                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                all_conds = input_data[1]
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * ["longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"])
                    uc = {"c_concat": all_conds["c_concat"], "c_crossattn": [uc]}
                samples_ddim, _ = sampler.module.sample(S=opt.ddim_steps,
                                                conditioning=all_conds,
                                                batch_size=batch_size,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=opt.scale,
                                                unconditional_conditioning=uc,
                                                eta=opt.ddim_eta,
                                                x_T=start_code)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                metrics = metrics_calculator.calc_metrics(data, x_samples_ddim,metrics=opt.metrics)
                
                for key in all_metrics_results.keys():
                    if len(all_metrics_results[key].keys()):
                        for metric_key in all_metrics_results[key].keys():
                            all_metrics_results[key][metric_key].append(metrics[key][metric_key])
                        with open(os.path.join(outpath,key+".csv"),"a") as f:
                            f.write(",".join(str(v[-1]) for v in all_metrics_results[key].values())+"\n")
                    else:
                        for metric_key in metrics[key].keys():
                            all_metrics_results[key][metric_key]=[metrics[key][metric_key]]
                        with open(os.path.join(outpath,key+".csv"),"w") as f:
                            f.write(",".join(str(v) for v in all_metrics_results[key].keys())+"\n")
                        with open(os.path.join(outpath,key+".csv"),"a") as f:
                            f.write(",".join(str(v[-1]) for v in all_metrics_results[key].values())+"\n")   

                x_samples=(x_samples_ddim*255).astype(np.uint8)

                pose_images = (einops.rearrange(model.pose_model(data["pose"]), 'b c h w -> b h w c') * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
                
                text_images = log_txt_as_img((x_samples.shape[1], x_samples.shape[2]), \
                                            [prompt_i+"\n"+",".join(str(v[-1]) for v in all_metrics_results["pose"].values()) for prompt_i in data["prompt"]], \
                                            size=x_samples.shape[2] // 25)
                text_images = (einops.rearrange(text_images, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                
                original_images=torch.clamp((data["jpg"]+ 1.0) / 2.0, min=0.0, max=1.0)
                original_images = original_images.cpu().numpy()* 255
                for batch_i in range(batch_size):
                    present_generated_img=x_samples[batch_i,...][:,:,[2,1,0]]
                    present_pose_image=pose_images[batch_i,...]
                    present_text_image=text_images[batch_i,...]
                    original_image=original_images[batch_i,...]
                    save_image=np.concatenate([present_generated_img,present_pose_image,present_text_image,original_image],1)
                    
                    save_path=os.path.join(outpath,data["img_path"][batch_i])
                    
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    
                    cv2.imwrite(save_path,save_image)
                
            # print
                        
            print(f"======================== batch:{batch_idx} ==========================")
            print("Present Metrics:")
            for key in all_metrics_results.keys():
                print(f"\t{key}:")
            
                for metric_key in all_metrics_results[key].keys():
                    print(f"\t{metric_key}: {list(all_metrics_results[key][metric_key])[-1]}")
                        
            print("==================================================")  
if __name__ == "__main__":
    main()

