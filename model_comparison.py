
# %%
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import h5py

from utils.create_data import sample_check
from utils.tools import random_bbox, mask_image, get_config
from model.networks import Generator

# Parameters
img_idx = 100
plt_scale = 0.1
rng = np.random.default_rng(0)
path_orig = r'C:/Users/s174370/Desktop/magfield-prediction/checkpoints/boundary_1_256/'

models = ['in_94_coarseG_l1', 'in_94_l1', 'in_94_lightweight']
# model = 'in_div_curl_1_94_1' #does not work with this script
file = h5py.File('data/bnd_256/magfield_256_large.h5') 

# Empty matrix so append errors (3 models, 5 performance tests: mse, psnr, mape, divergence, curl)
err_mat = np.empty([5,4])
# %%
for i in models:
    model = i
    exp_path = path_orig + model

    # Matrices for storing errors for the samples
    mse_mat = np.empty([1,img_idx])
    psnr_mat = np.empty([1,img_idx])
    mape_mat = np.empty([1,img_idx])
    div_mat = np.empty([1,img_idx])
    curl_mat = np.empty([1,img_idx])

    for j in np.arange(0, img_idx):
        # print(file['field'][img_idx,:,:,:,1].shape)
        field = file['field'][j,:,:,:,1]
        # Plot field chosen
        # sample_check(field, v_max=plt_scale, filename = 'orig_'+ model)
        
        # Make box 
        config = get_config(exp_path + '/config.yaml')
        bboxes = random_bbox(config, rng=rng)
        x, mask, orig = mask_image(np.array([field]), bboxes, config, bnd=config['boundary'])
        # print(x.shape)
        # Plot box made
        # sample_check(x[0], v_max=plt_scale, filename = 'boundary_'+model)

        # Test last generator ran
        last_model_name = exp_path + '/gen_00400000.pt'
        netG = Generator(config['netG'], config['coarse_G'], True, [0])
        netG.load_state_dict(torch.load(last_model_name))
        netG = nn.parallel.DataParallel(netG, device_ids=[0])
        corrupt_t = torch.from_numpy(x[0].astype('float32')).cuda().unsqueeze(0)
        mask_t = torch.from_numpy(mask[0].astype('float32')).cuda().unsqueeze(0)

        # Inference
        _, out = netG(corrupt_t, mask_t)

        # Plot original box (input)
        # sample_check(orig[0], v_max=plt_scale, filename = 'orig_box_'+model)

        out_np = out.squeeze(0).cpu().data.numpy()
        # Plot output
        # sample_check(out_np, v_max=plt_scale, filename='wgan_'+model)

        # Calculate performance of models
        diff = orig - out_np
        # mse_final = np.mean(diff**2)
        mse_mat[:,j] = np.mean(diff**2)
        if mse_mat[:,j] < 1e-4:
            psnr_mat[:,j] = 0
        else:
            # psnr = 20 * np.log10(np.max(orig) / np.sqrt(mse_final))
            psnr_mat[:,j] = 20 * np.log10(np.max(orig) / np.sqrt(mse_mat[:,j]))
        # mape = 100*(np.abs(np.mean(diff)/np.mean(orig)))
        mape_mat[:,j] = 100*(np.abs(np.mean(diff)/np.mean(orig)))

        # print(f"Recon loss: {np.mean(np.abs(diff)):.4f}")
        # print(f"PSNR: {psnr:.4f} dB")
        # print(f"MAPE: {mape:.4f} %")

        out_stack = torch.from_numpy(out_np)[None, :]

        # Div
        Hx_x = torch.gradient(out_stack[0,0], dim=1, edge_order=2)[0]
        Hy_y = torch.gradient(out_stack[0,1], dim=0, edge_order=2)[0]
        if len(out_stack.size()[1:]) > 3 : 
            Hz_z = torch.gradient(out_stack[0,2], dim=2, edge_order=2)[0]
            div_mag = torch.stack([Hx_x, Hy_y, Hz_z], dim=0)[:,:,:,1]
        else:
            div_mag = torch.stack([Hx_x, Hy_y], dim=0)
        # div = torch.mean(torch.abs(div_mag.sum(dim=0)))
        div_mat[:,j] = torch.mean(torch.abs(div_mag.sum(dim=0)))

        #Curl
        Hx_y = torch.gradient(out_stack[0,0], dim=0, edge_order=2)[0]
        Hy_x = torch.gradient(out_stack[0,1], dim=1, edge_order=2)[0]
        if len(out_stack.size()[1:]) > 3 :
            Hx_z = torch.gradient(out_stack[0,0], dim=2, edge_order=2)[0]
            Hy_z = torch.gradient(out_stack[0,1], dim=2, edge_order=2)[0]
            Hz_x = torch.gradient(out_stack[0,2], dim=1, edge_order=2)[0]
            Hz_y = torch.gradient(out_stack[0,2], dim=0, edge_order=2)[0]
            curl_vec = torch.stack([Hz_y - Hy_z, Hx_z - Hz_x, Hy_x - Hx_y], dim=0)[:,:,:,1]
            curl_mag = curl_vec.square().sum(dim=0)
        else:
            curl_mag = (Hy_x - Hx_y).square()
        # curl = torch.mean(curl_mag)
        curl_mat[:,j] = torch.mean(curl_mag)
        # print(f"divergence: {div:.5f}")
        # print(f"curl: {curl:.5f}")
    
    psnr_mat = psnr_mat[~np.isnan(psnr_mat)]
    
    err_mat[0,models.index(i)+1] = np.average(mse_mat)*1e3
    err_mat[1,models.index(i)+1] = np.average(psnr_mat)
    err_mat[2,models.index(i)+1] = np.average(mape_mat)
    err_mat[3,models.index(i)+1] = np.average(div_mat)*1e3
    err_mat[4,models.index(i)+1] = np.average(curl_mat)*1e6

#%%
err_list = err_mat.tolist()

err_list[0][0] = 'MSE [mT]'
err_list[1][0] = 'PSNR [dB]'
err_list[2][0] = 'MAPE [%]'
err_list[3][0] = 'Div [mT/px]'
err_list[4][0] = 'Curl [\u03BC T/px]'
print(tabulate(err_list, headers=['Test']+models, tablefmt="grid", showindex=False))


# %%
