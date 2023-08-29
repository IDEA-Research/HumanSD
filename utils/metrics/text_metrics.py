import torch
import numpy as np

from torchmetrics.multimodal import CLIPScore


class TextMetrics():
    def __init__(self,device,clip_similarity_score_model_name):
        self.device=device
        self.clip_similarity_score_model_name=clip_similarity_score_model_name
        self.clip_similarity_score_model=CLIPScore(model_name_or_path=clip_similarity_score_model_name).to(self.device)
        

    def clip_similarity_score(self,img, texts):
        return self.clip_similarity_score_model(img, texts)


    def compute(self,
                batch, 
                output_images):    
        if  type(output_images) is np.ndarray:
            output_images=torch.tensor(output_images)
            
        if  output_images.shape[-1]==3:
            output_images=output_images.permute(0,3,1,2)
            
        with torch.no_grad():
            clip_similarity_value=self.clip_similarity_score(output_images,batch["txt"])
        
        clip_similarity_result={
                "CLIP Similarity Score  (CLIPSIM)                                                  ":clip_similarity_value,
            }
        
        results={**clip_similarity_result}
        
        return results
        
    def __call__(self,batch, output_images):
        text_result=self.compute(batch, output_images)
        return text_result
    
#----------------------------------------------------------------------------
