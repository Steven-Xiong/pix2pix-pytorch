from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms

from utils import is_image_file, load_img, save_img

import lpips
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim
import numpy as np
import argparse
import cv2
from scipy.spatial import distance as dist
from scipy.stats import wasserstein_distance
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
from dataset import *
from pytorch_msssim import ssim

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--direction', type=str, default='b2a', help='a2b or b2a')
parser.add_argument('--nepochs', type=int, default=30, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument('--input_path', type=str, default = 'results/brooklyn_input_11.6')
parser.add_argument('--output_path', type= str, default = 'results/brooklyn_output_11.6')
parser.add_argument('--save_name', type = str, default='brooklyn_11.6')
opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)

transform_list = [transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

transform = transforms.Compose(transform_list)

#加test
testing_data_loader = DataLoader(
    brooklynqueensdataset(mode="test"),
    batch_size=opt.test_batch_size,
    shuffle=True,
    num_workers=4,
)
print('test:',len(testing_data_loader)*opt.test_batch_size)

from torch.autograd import Variable
Tensor = torch.cuda.FloatTensor 

def sd_func(real, fake):
    '''
    ref: page 6 in https://arxiv.org/abs/1511.05440
    '''
    dgt1 = torch.abs(torch.diff(real,dim=-2))[:, :, 1:, 1:-1]
    dgt2 = torch.abs(torch.diff(real, dim=-1))[:, :, 1:-1, 1:]
    dpred1 = torch.abs(torch.diff(fake, dim=-2))[:, :, 1:, 1:-1]
    dpred2 = torch.abs(torch.diff(fake, dim=-1))[:, :, 1:-1, 1:]
    return 10*torch.log10(1.**2/torch.mean(torch.abs(dgt1+dgt2-dpred1-dpred2))).cpu().item()

psnr_scores = []
ssim_scores = []
rmse_scores = []
sd_scores = []
lpips_scores = []
results_list = []
input_list = []
loss_fn_alex = lpips.LPIPS(net='alex').cuda() # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # closer to "traditional" perceptual loss, when used for optimization

input_path = opt.input_path
output_path = opt.output_path
save_name = opt.save_name
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

for i, batch in enumerate(testing_data_loader):
    # input, target = batch[0].to(device), batch[1].to(device)
    # if i>10:
    #     break
    input = Variable(batch["B"].type(Tensor))
    target = Variable(batch["A"].type(Tensor))
    
    # if i % 5 == 0:  # save images to an HTML file
    #     print('processing (%04d)-th image... %s' % (i, img_path))
    # save_images(webpage, visuals, img_path,id=i,aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        
    prediction = net_g(input).detach()
    img1 = target.squeeze(0)
    img2 = prediction.squeeze(0)
    
    #计算指标
    # visuals['fake_B_final'].shape[1,3,h,w] data['target'].shape:[1,3,h,w]
    # import pdb; pdb.set_trace()
    input_list.append(img1.unsqueeze(0))
    results_list.append(img2.unsqueeze(0))


#import pdb; pdb.set_trace()
print(len(input_list), len(results_list))
#rmse = np.sqrt(mean_squared_error(input_list, results_list))
mseloss = torch.nn.MSELoss(True,True)

for img1, img2 in zip(input_list, results_list):
    # import pdb; pdb.set_trace()
    psnr_tmp = -10*mseloss(img1,img2).log10().item()
    ssim_tmp = ssim(img1, img2,data_range=1.).item()
    # psnr_tmp = compute_psnr(img1,img2)
    # ssim_tmp = compute_ssim(img1,img2, channel_axis=-1, data_range=1.0) #或者ssim_tmp = compute_ssim(img1*255,img2*255, channel_axis=-1, data_range=255)
    # mse = np.mean((img1 - img2) ** 2)
    # rmse_tmp = np.sqrt(mse)
    
    #11.2改
    rmse_tmp = torch.sqrt(mseloss(img1*255.,img2*255.)).item()
    sd_tmp = sd_func(img1,img2)  
        
    lpips_tmp = torch.mean(loss_fn_alex(img1.to(device),img2.to(device))).cpu()
    
    psnr_scores.append(psnr_tmp)
    ssim_scores.append(ssim_tmp)
    rmse_scores.append(rmse_tmp)
    sd_scores.append(sd_tmp)
    lpips_scores.append(float(lpips_tmp))
    with open('evaluate_'+save_name+'_psnr.txt','a') as f:
        f.write(str(psnr_tmp)+'\n')
    with open('evaluate_'+save_name+'_ssim.txt','a') as f:
        f.write(str(ssim_tmp)+'\n')
    with open('evaluate_'+save_name+'_rmse.txt','a') as f:
        f.write(str(rmse_tmp)+'\n')
    with open('evaluate_'+save_name+'_sd.txt','a') as f:
        f.write(str(sd_tmp)+'\n')
    with open('evaluate_'+save_name+'_lpips.txt','a') as f:
        f.write(str(lpips_tmp)+'\n')

#import pdb; pdb.set_trace()
psnr_score = np.mean(psnr_scores)
ssim_score = np.mean(ssim_scores)
rmse_score = np.mean(rmse_scores)
sd_score = np.mean(sd_scores)
lpips_score = np.mean(lpips_scores)
#lpips_score = loss_fn_vgg(input_list, results_list)
#ssim_val = ssim( input_list, results_list, data_range=255, size_average=False)


for i, img_tensor in enumerate(input_list):
    #cv2.imwrite(os.path.join(input_path,('input'+str(i)+'.png')), img_tensor*255)
    # import pdb; pdb.set_trace()
    transforms.ToPILImage()(img_tensor[0]).save(os.path.join(input_path,(str(i)+'.png')))
for i, img_tensor in enumerate(results_list):
    #cv2.imwrite(os.path.join(output_path,('output'+str(i)+'.png')), img_tensor*255)
    transforms.ToPILImage()(img_tensor[0]).save(os.path.join(output_path,(str(i)+'.png')))
#import pdb; pdb.set_trace()
    
fid = fid_score.calculate_fid_given_paths([input_path, output_path], 
                                                batch_size=50, 
                                                device=device, 
                                                dims=2048)
print('PSNR: ', psnr_score, 'SSIM: ', ssim_score, 'FID: ', fid, 'RMSE: ', rmse_score, 'SD: ', sd_score,'LPIPS: ', lpips_score)
with open('evaluate_'+save_name+'_.txt','a') as f:
    f.write('PSNR: '+ str(psnr_score))
    f.write('\nSSIM: '+ str(ssim_score))
    f.write('\nRMSE: '+ str(rmse_score))
    f.write('\nFID: '+ str(fid))
    f.write('\nSD: '+ str(sd_score))
    f.write('\nlpips: '+ str(lpips_score))
    