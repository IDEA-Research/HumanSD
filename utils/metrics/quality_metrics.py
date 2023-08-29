import numpy as np
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.utilities.data import dim_zero_cat

from PIL import Image
import torchvision.transforms as TF
from tqdm import tqdm
import json
import os

class QualityMetrics():
    def __init__(self,
                 device,
                 refer_dataset_base_dir,
                 refer_dataset_json_path,
                 fid_model_feature,
                 kid_subset_size):
        
        # FID
        self.refer_dataset_base_dir=refer_dataset_base_dir
        self.refer_dataset_json_path=refer_dataset_json_path
        self.device=device
        
        fid_image_transforms=TF.Compose([
                TF.Resize(299),
                TF.CenterCrop(299),
                TF.ToTensor(),
                TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        dataset_imgs=[]
        with open(refer_dataset_json_path,"r") as f:
            dataset_json=json.load(f)
        
        print("initialize contrast dataset")
        for image_i in tqdm(range(len(dataset_json["images"]))):
            present_image_path = os.path.join(refer_dataset_base_dir,dataset_json["images"][image_i]["file_name"])
            img = Image.open(present_image_path).convert('RGB')
            dataset_imgs.append(fid_image_transforms(img).unsqueeze(0))
        
        dataset_imgs=torch.concat(dataset_imgs).to(self.device)
        
        self.fid_model_feature=fid_model_feature
        self.fid_model=FrechetInceptionDistance(feature=self.fid_model_feature,normalize=True).to(self.device)
        self.fid_model.update(dataset_imgs,real=True)
        
        # IS
        self.inception_model = InceptionScore(normalize=True).to(self.device)
        
        # KID
        self.kid_subset_size=kid_subset_size
        self.kid_model = KernelInceptionDistance(kid_subset_size=self.kid_subset_size,normalize=True).to(self.device)
        self.kid_model.update(dataset_imgs, real=True)
        

    def calculate_fid(self, img):
        self.fid_model.update(img, real=False)
        return self.fid_model.compute()
    
    def calculate_kid(self, img):
        self.kid_model.update(img, real=False)
        if dim_zero_cat(self.kid_model.fake_features).shape[0]<=self.kid_model.subset_size:
            return None
        return self.kid_model.compute()

    def calculate_is(self,img):
        self.inception_model.update(img)
        return self.inception_model.compute()

    def compute(self, batch, output_images):    
        if  type(output_images) is np.ndarray:
            output_images=torch.tensor(output_images)
            
        if  output_images.shape[-1]==3:
            output_images=output_images.permute(0,3,1,2)
            
        with torch.no_grad():
            fid_value=self.calculate_fid(output_images)
            is_value=self.calculate_is(output_images)
            kid_value=self.calculate_kid(output_images)
        
        fid_result={
                "Fréchet Inception Distance    (FID)                                               ": fid_value,
            }
        
        is_result={
                "Inception Score (IS)                                                              ": is_value,
            }
        
        kid_result={
                "Kernel Inception Distance (KID)                                                    ": kid_value,
            }
        
        results={**fid_result,**is_result,**kid_result}
        
        return results
    
    def __call__(self,batch, output_images):
        quality_result=self.compute(batch, output_images)
        return quality_result
        

#----------------------------------------------------------------------------

    

# """Calculates the Frechet Inception Distance (FID) to evalulate GANs

# The FID metric calculates the distance between two distributions of images.
# Typically, we have summary statistics (mean & covariance matrix) of one
# of these distributions, while the 2nd distribution is given by a GAN.

# When run as a stand-alone program, it compares the distribution of
# images that are stored as PNG/JPEG at a specified location with a
# distribution given by summary statistics (in pickle format).

# The FID is calculated by assuming that X_1 and X_2 are the activations of
# the pool_3 layer of the inception net for generated samples and real world
# samples respectively.

# See --help to see further details.

# Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
# of Tensorflow

# Copyright 2018 Institute of Bioinformatics, JKU Linz

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# """
# import sys

# import numpy as np
# import torch
# import torchvision.transforms as TF
# from PIL import Image
# from scipy import linalg
# from torch.nn.functional import adaptive_avg_pool2d

# from sklearn.metrics.pairwise import polynomial_kernel

# try:
#     from tqdm import tqdm
# except ImportError:
#     # If tqdm is not available, provide a mock version of it
#     def tqdm(x):
#         return x

# from pytorch_fid.inception import InceptionV3

# IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
#                     'tif', 'tiff', 'webp'}


# class ImagePathDataset(torch.utils.data.Dataset):
#     def __init__(self, files, transforms=None):
#         self.files = files
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, i):
#         path = self.files[i]
#         img = Image.open(path).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img

# class ImageDataset(torch.utils.data.Dataset):
#     def __init__(self, images, transforms=None):
#         self.images = images
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, i):
#         img = self.images[i]
#         img = Image.fromarray((img*255).astype(np.uint8)).convert('RGB')
#         if self.transforms is not None:
#             img = self.transforms(img)
#         return img


# def get_activations(files, model,image_dataset, batch_size=50, dims=2048, device='cpu',
#                     num_workers=1):
#     """Calculates the activations of the pool_3 layer for all images.

#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : Batch size of images for the model to process at once.
#                      Make sure that the number of samples is a multiple of
#                      the batch size, otherwise some samples are ignored. This
#                      behavior is retained to match the original FID score
#                      implementation.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- A numpy array of dimension (num images, dims) that contains the
#        activations of the given tensor when feeding inception with the
#        query tensor.
#     """
#     model.eval()

#     if batch_size > len(files):
#         print(('Warning: batch size is bigger than the data size. '
#                'Setting batch size to data size'))
#         batch_size = len(files)

#     dataset = image_dataset(files, transforms=TF.Compose([
#                 TF.Resize(299),
#                 TF.CenterCrop(299),
#                 TF.ToTensor(),
#                 TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]))
#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              drop_last=False,
#                                              num_workers=num_workers)

#     pred_arr = np.empty((len(files), dims))

#     start_idx = 0

#     for batch in tqdm(dataloader):
#         batch = batch.to(device)

#         with torch.no_grad():
#             pred = model(batch)[0]

#         # If model output is not scalar, apply global spatial average pooling.
#         # This happens if you choose a dimensionality not equal 2048.
#         if pred.size(2) != 1 or pred.size(3) != 1:
#             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

#         pred = pred.squeeze(3).squeeze(2).cpu().numpy()

#         pred_arr[start_idx:start_idx + pred.shape[0]] = pred

#         start_idx = start_idx + pred.shape[0]

#     return pred_arr


# def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     """Numpy implementation of the Frechet Distance.
#     The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
#     and X_2 ~ N(mu_2, C_2) is
#             d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

#     Stable version by Dougal J. Sutherland.

#     Params:
#     -- mu1   : Numpy array containing the activations of a layer of the
#                inception net (like returned by the function 'get_predictions')
#                for generated samples.
#     -- mu2   : The sample mean over activations, precalculated on an
#                representative data set.
#     -- sigma1: The covariance matrix over activations for generated samples.
#     -- sigma2: The covariance matrix over activations, precalculated on an
#                representative data set.

#     Returns:
#     --   : The Frechet Distance.
#     """

#     mu1 = np.atleast_1d(mu1)
#     mu2 = np.atleast_1d(mu2)

#     sigma1 = np.atleast_2d(sigma1)
#     sigma2 = np.atleast_2d(sigma2)

#     assert mu1.shape == mu2.shape, \
#         'Training and test mean vectors have different lengths'
#     assert sigma1.shape == sigma2.shape, \
#         'Training and test covariances have different dimensions'

#     diff = mu1 - mu2

#     # Product might be almost singular
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if not np.isfinite(covmean).all():
#         msg = ('fid calculation produces singular product; '
#                'adding %s to diagonal of cov estimates') % eps
#         print(msg)
#         offset = np.eye(sigma1.shape[0]) * eps
#         covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

#     # Numerical error might give slight imaginary component
#     if np.iscomplexobj(covmean):
#         if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
#             m = np.max(np.abs(covmean.imag))
#             raise ValueError('Imaginary component {}'.format(m))
#         covmean = covmean.real

#     tr_covmean = np.trace(covmean)

#     return (diff.dot(diff) + np.trace(sigma1)
#             + np.trace(sigma2) - 2 * tr_covmean)


# def calculate_activation_statistics(files, model, image_dataset,batch_size=50, dims=2048,
#                                     device='cpu', num_workers=1):
#     """Calculation of the statistics used by the FID.
#     Params:
#     -- files       : List of image files paths
#     -- model       : Instance of inception model
#     -- batch_size  : The images numpy array is split into batches with
#                      batch size batch_size. A reasonable batch size
#                      depends on the hardware.
#     -- dims        : Dimensionality of features returned by Inception
#     -- device      : Device to run calculations
#     -- num_workers : Number of parallel dataloader workers

#     Returns:
#     -- mu    : The mean over samples of the activations of the pool_3 layer of
#                the inception model.
#     -- sigma : The covariance matrix of the activations of the pool_3 layer of
#                the inception model.
#     """
#     act = get_activations(files, model, image_dataset,batch_size, dims, device, num_workers)
#     mu = np.mean(act, axis=0)
#     sigma = np.cov(act, rowvar=False)
#     return mu, sigma, act

# def _sqn(arr):
#     flat = np.ravel(arr)
#     return flat.dot(flat)

# def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
#                        mmd_est='unbiased', block_size=1024,
#                        var_at_m=None, ret_var=True):
#     # based on
#     # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
#     # but changed to not compute the full kernel matrix at once
#     m = K_XX.shape[0]
#     assert K_XX.shape == (m, m)
#     assert K_XY.shape == (m, m)
#     assert K_YY.shape == (m, m)
#     if var_at_m is None:
#         var_at_m = m

#     # Get the various sums of kernels that we'll use
#     # Kts drop the diagonal, but we don't need to compute them explicitly
#     if unit_diagonal:
#         diag_X = diag_Y = 1
#         sum_diag_X = sum_diag_Y = m
#         sum_diag2_X = sum_diag2_Y = m
#     else:
#         diag_X = np.diagonal(K_XX)
#         diag_Y = np.diagonal(K_YY)

#         sum_diag_X = diag_X.sum()
#         sum_diag_Y = diag_Y.sum()

#         sum_diag2_X = _sqn(diag_X)
#         sum_diag2_Y = _sqn(diag_Y)

#     Kt_XX_sums = K_XX.sum(axis=1) - diag_X
#     Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
#     K_XY_sums_0 = K_XY.sum(axis=0)
#     K_XY_sums_1 = K_XY.sum(axis=1)

#     Kt_XX_sum = Kt_XX_sums.sum()
#     Kt_YY_sum = Kt_YY_sums.sum()
#     K_XY_sum = K_XY_sums_0.sum()

#     if mmd_est == 'biased':
#         mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
#                 + (Kt_YY_sum + sum_diag_Y) / (m * m)
#                 - 2 * K_XY_sum / (m * m))
#     else:
#         assert mmd_est in {'unbiased', 'u-statistic'}
#         mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
#         if mmd_est == 'unbiased':
#             mmd2 -= 2 * K_XY_sum / (m * m)
#         else:
#             mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

#     if not ret_var:
#         return mmd2

#     Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
#     Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
#     K_XY_2_sum = _sqn(K_XY)

#     dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
#     dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

#     m1 = m - 1
#     m2 = m - 2
#     zeta1_est = (
#         1 / (m * m1 * m2) * (
#             _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
#         - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
#         + 1 / (m * m * m1) * (
#             _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
#         - 2 / m**4 * K_XY_sum**2
#         - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
#         + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
#     )
#     zeta2_est = (
#         1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
#         - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
#         + 2 / (m * m) * K_XY_2_sum
#         - 2 / m**4 * K_XY_sum**2
#         - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
#         + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
#     )
#     var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
#                + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

#     return mmd2, var_est


# def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
#                    var_at_m=None, ret_var=True):
#     # use  k(x, y) = (gamma <x, y> + coef0)^degree
#     # default gamma is 1 / dim
#     X = codes_g
#     Y = codes_r

#     K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
#     K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
#     K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

#     return _mmd2_and_variance(K_XX, K_XY, K_YY,
#                               var_at_m=var_at_m, ret_var=ret_var)


# def polynomial_mmd_averages(codes_g, codes_r, subset_size=1000,
#                             ret_var=False, **kernel_args):
#     m = min(codes_g.shape[0], codes_r.shape[0])

#     g = codes_g[np.random.choice(len(codes_g), subset_size, replace=False)]
#     r = codes_r[np.random.choice(len(codes_r), subset_size, replace=False)]
#     o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
#     if ret_var:
#         mmds, vars = o
#     else:
#         mmds = o

#     return (mmds, vars) if ret_var else mmds


# class quality_calculate():
#     def __init__(self,dataset_img_paths, dims, device, batch_size, num_workers=1,subset_size=100):
        
#         self.device = device
#         self.dims = dims
#         self.batch_size = batch_size
#         self.dataset_img_paths = dataset_img_paths
#         self.num_workers = num_workers
#         block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

#         self.model = InceptionV3([block_idx]).to(device)
#         self.model.eval
        
#         self.m1, self.s1, self.act1 = calculate_activation_statistics(dataset_img_paths, self.model, ImagePathDataset, batch_size,
#                                                dims, device, num_workers)
        
#         self.subset_size=subset_size    

#     def calculate(self, generated_images):
        
#         if hasattr(self, 'generated_images'):
#             self.generated_images=np.concatenate((self.generated_images, generated_images),0)
#         else:
#             self.generated_images=generated_images


    
#     def summarize_fid(self):
        
#         with torch.no_grad():
#             m2, s2,_ = calculate_activation_statistics(self.generated_images, self.model, ImageDataset, self.batch_size,
#                                                self.dims, self.device, self.num_workers)
        
#             fid_value = calculate_frechet_distance(self.m1, self.s1, m2, s2)
        
#         return fid_value
    
    
#     def summarize_kid(self):
        
#         with torch.no_grad():
#             act2 = get_activations(self.generated_images, self.model, ImageDataset, self.batch_size,
#                                                self.dims, self.device, self.num_workers)
        
#             kid_value = polynomial_mmd_averages(self.act1, act2, subset_size=self.subset_size)
        
#         return kid_value


#     def compute(self,generated_images):
        
#         self.calculate(generated_images)
        
#         if self.generated_images.shape[0]>=self.subset_size:
#             fid_value=self.summarize_fid()
            
#             kid_value=self.summarize_kid()

#             fid_result={
#                     "Fréchet Inception Distance    (FID)                                               ": fid_value,
#             }
#             kid_result={
#                     "Kernel Inception Distance    (KID)                                               ": kid_value
#             }
        
#         else:
#             fid_result={
#                     "Fréchet Inception Distance    (FID)                                               ": None
#             }
#             kid_result={
#                     "Kernel Inception Distance    (KID)                                               ": None
#             }
            
#         results={**fid_result,**kid_result}
            
#         return results