import os
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from scipy.interpolate import griddata, bisplrep, bisplev
from skimage.restoration import inpaint
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from model.networks import Generator
from utils.tools import calc_div, calc_curl, random_bbox, mask_image
from utils.tools import field_loader, get_config, get_model_list

parser = ArgumentParser()
parser.add_argument('--name', type=str, default='in_ext_div_curl_1_144_1', 
    help='manual seed')
parser.add_argument('--exp', type=str, default='paper',
    help='manual seed')
parser.add_argument('--cfg_file', type=str, default='test.yaml',
    help='Path to test configuration')
parser.add_argument('--num_samples', type=int, default=5,
    help='Number of samples to predict')
parser.add_argument('--box_amount', type=int, default=None,
    help='Number of bounding boxes as mask')
parser.add_argument('--mask_size', type=int, default=None,
    help='Size of the bounding boxes')
parser.add_argument('--err_scale', type=float, default=None,
    help='Scale of error colorbar')
parser.add_argument('--method', type=str, default='wgan',
    help='Method to use for predicting missing values')
parser.add_argument('--lab', type=bool, default=False,
    help='Use lab setup')
parser.add_argument('--plot', type=bool, default=True,
    help='Plot results')
parser.add_argument('--save', type=bool, default=False,
    help='Save results as pandas DataFrame')


def predict(
    name,
    exp='foo',
    cfg_file='test.yaml',
    num_samples=100,
    plot=False,
    err_min=0,
    err_max=256,
    err_scale=0.01,
    plot_scale=0.05,
    box_amount=None,
    mask_size=None,
    mask_distributed=False,
    lab=False,
    eval_idx=None,
    seed=0,
    method='wgan',
    save=False
):
    checkpoint_path = Path(__file__).parent.resolve() / 'checkpoints' / exp / name
    output_path = checkpoint_path / 'test'
    if not output_path.exists(): output_path.mkdir()

    df_px = pd.DataFrame([], columns=['i', 'j', 'd_h', 'd_w', 'err', 'err_pct'])
    df_mask = pd.DataFrame([], columns=['d_h', 'd_w', 'err', 'err_pct'])
    df_eval = pd.DataFrame([], columns=['loss', 'loss_pct', 'div', 'curl'])

    for i in range(num_samples):
        cfg_path=Path(__file__).parent.resolve() / 'configs' / cfg_file
        config = get_config(cfg_path)
        # Setting additional configuration
        if box_amount is not None:
            config['box_amount'] = box_amount
        if mask_size is not None:
            config['mask_shape'] = [mask_size, mask_size]

        # CUDA configuration
        cuda = config['cuda']
        device_ids = config['gpu_ids']
        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
            device_ids = list(range(len(device_ids)))
            config['gpu_ids'] = device_ids
            cudnn.benchmark = True

        # Setting seeds for reproducability
        torch.manual_seed(seed)
        if cuda: torch.cuda.manual_seed_all(seed)

        # Magnetic field paths and index control for sample
        if lab:
            data_idx = 0
        elif eval_idx is None:
            data_idx = i
        else:
            data_idx = eval_idx
        magfield_path = Path(__file__).parent.resolve() / 'data' \
            / config["val_data"] / f'{data_idx}.npy'

        with torch.no_grad():
            gt = field_loader(magfield_path, factor=config['scale_factor'], lab=lab)
            gt_top = None
            gt_bottom = None
            if not lab:
                gt_top = torch.from_numpy(gt[:,:,:,:,0].astype('float32'))
                gt_bottom = torch.from_numpy(gt[:,:,:,:,2].astype('float32'))
                gt = torch.from_numpy(gt[:,:,:,:,1].astype('float32'))

            bboxes = random_bbox(config, distributed=mask_distributed, seed=i)
            x, mask = mask_image(gt, bboxes, config)

            # Prediction of missing field data with different methods
            if method == 'wgan':
                # Load generator network with parameters of best trained model
                netG = Generator(config['netG'], cuda, device_ids)
                if config['resume'] is None:
                    last_model_name = get_model_list(checkpoint_path,
                                                     "gen", best=True)
                else:
                    last_model_name = get_model_list(checkpoint_path, "gen",
                                                     iteration=config['resume'])
                netG.load_state_dict(torch.load(last_model_name))

                if cuda:
                    netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                    x = x.cuda()
                    mask = mask.cuda()
                    if gt_top is not None:
                        gt_top = gt_top.cuda()
                        gt_bottom = gt_bottom.cuda()

                # Inference
                _, x2 = netG(x, mask)              

            elif method in ['linear', 'spline', 'gaussian']:
                # Splitting up for each spatial location
                x_post = np.zeros(x.shape)
                grid_x, grid_y = np.mgrid[0:config['image_shape'][1]:1,
                                          0:config['image_shape'][2]:1]
                if method == 'gaussian':
                    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))

                for l in range(config['image_shape'][0]):
                    points = np.concatenate(
                        (
                            np.expand_dims(np.where(mask[0,0] == 0)[0], axis=-1),
                            np.expand_dims(np.where(mask[0,0] == 0)[1], axis=-1)
                        ), axis=-1)
                    values = x[0,l,points[:,0],points[:,1]].cpu().data.numpy()
                    eval_pts = np.concatenate(
                        (
                            np.expand_dims(grid_x, axis=-1),
                            np.expand_dims(grid_y, axis=-1)
                        ), axis=-1).reshape((-1,2))

                    if method == 'linear':
                        x_post[0,l,:,:] = griddata(points, values, (grid_x, grid_y),
                                                   method='linear')

                    elif method == 'spline':
                        if config['outpaint']:
                            tck = bisplrep(
                                points[:,0], points[:,1], values,
                                xb=0, xe=config['image_shape'][1],
                                yb=0, ye=config['image_shape'][2]
                            )
                            x_post[0,l,:,:] = bisplev(
                                np.arange(config['image_shape'][1]),
                                np.arange(config['image_shape'][2]),
                                tck
                            )
                        else:
                            x_post[0,l,:,:] = griddata(points, values, (grid_x, grid_y),
                                                       method='cubic')

                    elif method == 'gaussian':
                        scaler = preprocessing.StandardScaler().fit(points)
                        pts_scaled = scaler.transform(points)
                        eval_pts_scaled = scaler.transform(eval_pts)
                        gpr = GPR(kernel=kernel, random_state=0).fit(pts_scaled, values)
                        x_post[0,l,:,:] = gpr.predict(eval_pts_scaled).reshape(
                            config['image_shape'][1], config['image_shape'][2])
                
                x2 = torch.from_numpy(x_post)

            elif method == 'biharmonic':
                x_pre = x.squeeze(0).permute((1,2,0)).cpu().data.numpy()
                mask_pre = mask.squeeze(0).squeeze(0).cpu().data.numpy()
                x_post = inpaint.inpaint_biharmonic(x_pre, mask_pre, multichannel=True)
                x2 = torch.from_numpy(x_post).permute((2,0,1)).unsqueeze(0)

            else:
                raise NotImplementedError(f'Method {method} is currently not supported')

            # Add prediction to known field points
            # Outpaint: Take whole prediction
            # Inpaint: Add predicted points only on missing areas
            if config['outpaint']:
                x2_eval = x2
            else:
                x2_eval = x2 * mask + x * (1. - mask)

            if not lab:
                field = torch.cat([
                    gt_top.unsqueeze(-1),
                    x2_eval.unsqueeze(-1),
                    gt_bottom.unsqueeze(-1)
                ], dim=-1)
            else:
                field = x2_eval
            
            div = calc_div(field, lab).cpu().data.numpy()
            curl = calc_curl(field, lab).cpu().data.numpy()
        
        gt = gt.cpu().data.numpy() / config['scale_factor']
        x2_eval = x2_eval.cpu().data.numpy() / config['scale_factor']
        masked_image = x.cpu().data.numpy() / config['scale_factor']

        # For lab setup only a smaller area is measured
        if not lab:
            err = abs(gt - x2_eval)
        else:
            err = abs(gt[:,:,:,:config['image_shape'][2]] \
                - x2_eval[:,:,:,:config['image_shape'][2]])
        
        # Calculate errors for masks in image
        if config['outpaint']:
            t_arr = bboxes[0,:,0].cpu().data.numpy()
            l_arr = bboxes[0,:,1].cpu().data.numpy()
            b_arr = t_arr + bboxes[0,:,2].cpu().data.numpy()
            r_arr = l_arr + bboxes[0,:,3].cpu().data.numpy()

            df_px = df_px.append([pd.DataFrame([
                [
                    k,
                    m,
                    min(np.min(np.abs(t_arr - k)), np.min(np.abs(b_arr - k))),
                    min(np.min(np.abs(l_arr - k)), np.min(np.abs(r_arr - k))),
                    np.mean(err[0,:,k,m]),
                    np.mean(err[0,:,k,m] / abs(gt[0,:,k,m]))
                ]
            ], columns=['i', 'j', 'd_h', 'd_w', 'err', 'err_pct'])
            for k in range(config['image_shape'][1])
            for m in range(config['image_shape'][2])
            if abs(gt[0,0,k,m]) != 0], ignore_index=True)

        else:
            for bbox in bboxes[0]:
                t, l, h, w = bbox.cpu().data.numpy()
                d_h = min(t, config['image_shape'][1] - (t + h))
                d_w = min(l, config['image_shape'][2] - (l + w))
                
                df_mask = df_mask.append(pd.DataFrame([[
                    d_h,
                    d_w,
                    err[0,:,t:t+h,l:l+w].mean(),
                    np.mean(err[0,:,t:t+h,l:l+w] / abs(gt[0,:,t:t+h,l:l+w]))
                ]], columns=['d_h', 'd_w', 'err', 'err_pct']), ignore_index=True)

                df_px = df_px.append([pd.DataFrame([
                    [
                        k,
                        m,
                        min(abs(t - k), abs(t + h - k)),
                        min(abs(l - m), abs(l + w - m)),
                        np.mean(err[0,:,k,m]),
                        np.mean(err[0,:,k,m] / abs(gt[0,:,k,m]))
                    ]
                ], columns=['i', 'j', 'd_h', 'd_w', 'err', 'err_pct'])
                for k in range(t, t + h)
                for m in range(l, l + w)
                ], ignore_index=True)

        # Mean pixel error
        part_err = err[0,:,err_min:err_max,err_min:err_max]
        if np.count_nonzero(part_err) == 0:
            loss = 0
        else:
            loss = np.sum(part_err) / np.count_nonzero(part_err)

        # Pixel-wise percentage error 
        pct = part_err / abs(gt[0,:,err_min:err_max,err_min:err_max])
        print(
            'Mean:', np.mean(pct[np.where(pct!=0)].flatten()),
            'Std:', np.std(pct[np.where(pct!=0)].flatten()) ,
            'Median:', np.median(pct[np.where(pct!=0)].flatten())
        )
        loss_pct = np.median(pct[np.where(pct!=0)].flatten())

        # Storing losses in DataFrame
        df_eval = df_eval.append(pd.DataFrame([
            [
                loss,
                loss_pct,
                div,
                curl
            ]
        ], columns=['loss', 'loss_pct', 'div', 'curl']), ignore_index=True)

        if plot:
            figpath = output_path / 'figs'
            if not figpath.exists(): figpath.mkdir()
            plt.close('all')
            fig, axes = plt.subplots(nrows=config['netG']['input_dim'],
                                     ncols=4, sharex=True, sharey=True)
            mode = 'Out' if config['outpaint'] else 'In'
            viz_list = [
                ('Truth_X', gt[0,0]),
                ('Masked_X', masked_image[0,0]),
                (f'{mode}paint_X', x2_eval[0,0]),
                ('Error_X', err[0,0]),
                ('Truth_Y', gt[0,1]),
                ('Masked_Y', masked_image[0,1]),
                (f'{mode}paint_Y', x2_eval[0,1]),
                ('Error_Y', err[0,1]),
                ('Truth_Z', gt[0,2]),
                ('Masked_Z', masked_image[0,2]),
                (f'{mode}paint_Z', x2_eval[0,2]),
                ('Error_Z', err[0,2])
            ]

            for j, (title, data) in enumerate(viz_list):
                ax = axes.flat[j]
                ax.set_title(title)

                if 'Error' in title:
                    cmap = 'cividis'
                    norm = colors.Normalize(vmin=0, vmax=err_scale)
                else:
                    cmap = 'bwr'
                    norm = colors.Normalize(vmin=-plot_scale, vmax=plot_scale)

                im = ax.imshow(data, cmap=cmap, norm=norm, origin="lower")

            fig.colorbar(im, ax=axes.ravel().tolist())
            figname = f'{config["box_amount"]}_{config["mask_shape"][0]}_{i}_{method}'
            plt.savefig(f'{figpath}/{figname}.png')
            # plt.savefig(f'{figpath}/{figname}.svg')

    if save:
        df_eval.attrs['name'] = name
        df_eval.attrs['box_amount'] = config['box_amount']
        df_eval.attrs['mask_shape'] = config['mask_shape'][0]
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        
        fname = str(config["box_amount"]) + '_' + str(config["mask_shape"][0]) \
            + '_' + timestamp + '_' + method
        df_eval.to_pickle(f'{output_path}/{fname}.p')
        df_px.to_pickle(f'{output_path}/{fname}_px.p')

        if not config['outpaint']:
            df_mask.to_pickle(f'{output_path}/{fname}_mask.p')


def eval_test(name, timestamp, exp='foo'):
    output_path = Path(__file__).parent.resolve() \
        / 'checkpoints' / exp / name / 'test'
    df = pd.read_pickle(f'{output_path}/{timestamp}.p')

    print(f'Summary: MAE: {df["loss"].mean():.6f} | [%]: {df["loss_pct"].mean()*100:.2f}' + \
          f' | Div: {df["div"].mean():.6f} | Curl: {df["curl"].mean():.6f}')


def plot_eval(
        name,
        box_amount=[1, 3, 5, 10, 20],
        mask_size=[4, 16, 32, 48, 64],
        exp='foo',
        mode='px',
        local=False,
        key='err',
        plot_overview=True,
        print_loss=False,
        methods=None,
        method_fixed=False,
    ):
    output_path = Path(__file__).parent.resolve() / 'checkpoints' / exp / name / 'test'
    plotpath = output_path / 'plots'
    if not plotpath.exists(): plotpath.mkdir()
    
    if methods is None:
        if name[:2] == 'in': 
            labels = ['WGAN-GP', 'Ours', 'Linear', 'Spline', 'Biharmonic']
            methods = ['wgan', 'wgan_phys', 'linear', 'spline', 'biharmonic']
        elif name[:3] == 'out':
            labels = ['WGAN-GP', 'Ours', 'Gaussian', 'Spline', 'Biharmonic']
            methods = ['wgan', 'wgan_phys', 'gaussian', 'spline', 'biharmonic']
    else:
        labels = methods

    dict_df = {}
    for m in methods:
        dict_df[m] = pd.DataFrame([],
            columns=['amount', 'size', 'mae', 'loss_pct', 'div', 'curl'])
    
    l_colors = ['b', 'k', 'r', 'y', 'g', 'm', 'o']

    for amount, size in list(itertools.product(box_amount, mask_size)):
        plt.close('all')
        ax = plt.gca()

        for j, method in enumerate(list(dict_df.keys())):
            try:
                if 'wgan' in method:
                    c_name = f'{name.split("_")[0]}_{name.split("_")[1]}_'
                    if 'phys' in method:
                        c_name += f'div_curl_{amount}_{size}'
                    else:
                        c_name += f'{amount}_{size}'
                    c_name += '_1'
                    o_path = Path(__file__).parent.resolve() \
                        / 'checkpoints' / exp / c_name / 'test'
                    method_name = 'wgan'
                else:
                    o_path = output_path
                    method_name = method
                
                if 'wgan' in method:
                    try:
                        if method_fixed: raise NameError('method_fixed')
                        df_a = pd.read_pickle(
                            sorted([
                                f for f in o_path.iterdir()
                                if f.is_file()
                                and f'{amount}_{size}_' in f.name
                                and ".p" in f.name
                                and not 'mask' in f.name
                                and not 'px' in f.name 
                                and method_name in f.name
                            ])[-1]
                        )
                    except:
                        print(f'Falling back to G trained on {name}')
                        o_path = output_path
                        df_a = pd.read_pickle(
                            sorted([
                                f for f in o_path.iterdir()
                                if f.is_file()
                                and f'{amount}_{size}_' in f.name
                                and ".p" in f.name
                                and not 'mask' in f.name
                                and not 'px' in f.name
                                and method_name in f.name
                            ])[-1]
                        )
                else:
                    df_a = pd.read_pickle(
                        sorted([
                            f for f in o_path.iterdir()
                            if f.is_file()
                            and f'{amount}_{size}_' in f.name
                            and ".p" in f.name
                            and not 'mask' in f.name
                            and not 'px' in f.name
                            and method_name in f.name
                        ])[-1]
                    )
                         
                df_a_px = pd.read_pickle(
                    sorted([
                        f for f in o_path.iterdir()
                        if f.is_file()
                        and f'{amount}_{size}_' in f.name
                        and "_px.p" in f.name
                        and method_name in f.name
                    ])[-1]
                )
                
                # Take the first 250 test for evaluation
                df_a = df_a.head(250)

                if name[:2] == 'in':
                    df_a_mask = pd.read_pickle(
                        sorted([
                            f for f in o_path.iterdir()
                            if f.is_file()
                            and f'{amount}_{size}_' in f.name
                            and "_mask.p" in f.name
                            and method_name in f.name
                        ])[-1]
                    )
                else:
                    df_a_mask = None
                
                dict_mode = {'px':df_a_px, 'mask':df_a_mask}

            except:
                print(f"Experiment {amount}_{size} with {method} has not been run!")
                exit()
            
            for _, row in df_a.iterrows():
                dict_df[method] = dict_df[method].append(
                    pd.DataFrame([[
                        amount,
                        size,
                        row['loss'],
                        row['loss_pct'],
                        row['div'],
                        row['curl'],
                    ]], columns=['amount', 'size', 'mae', 'loss_pct', 'div', 'curl']),
                    ignore_index=True
                )

            # Distance plots
            if local:
                dict_mode[mode] = dict_mode[mode].assign(
                    d = np.where(dict_mode[mode]['d_h'] < dict_mode[mode]['d_w'],
                        dict_mode[mode]['d_h'],
                        dict_mode[mode]['d_w']
                    )
                )
            else:
                dict_mode[mode] = dict_mode[mode].assign(
                    d = np.where(dict_mode[mode]['i'] < dict_mode[mode]['j'],
                        np.minimum(dict_mode[mode]['i'], 256-dict_mode[mode]['j']),
                        np.minimum(dict_mode[mode]['j'], 256-dict_mode[mode]['i'])
                    )
                )
            
            dict_mode[mode].groupby(['d']).mean().reset_index().plot.scatter(
                x='d', y=key, c=l_colors[j], ax=ax)
            labels.append(method)
        
        if key == 'err_pct':
            ax.set_ylim([0,0.30])
        if local:
            f_local = '_local'
            plt.xlabel('Distance to mask edge')            
        else:
            f_local = ''
            ax.set_ylim([0,0.05])
            plt.xlabel('Distance to image border')
            
        plt.legend(labels)
        if len(mask_size) > len(box_amount):
            plt.title(f'Single mask')
        else:
            plt.title(f'Multiple masks')

        plt.ylabel(key)
        figname = f'{amount}_{size}_{mode}{f_local}_{key}'
        plt.savefig(f'{output_path}/plots/{figname}.png')
        # plt.savefig(f'{output_path}/plots/{figname}.svg')
    
    if plot_overview:
        for metric in ['mae', 'loss_pct', 'div', 'curl']:
            plt.close('all')
            ax = plt.gca()
            if metric == 'div':
                ax.set_ylim([0,0.0003])
            elif metric == 'curl':
                ax.set_ylim([0,3e-6])
            
            if len(mask_size) > len(box_amount):
                group_idx = 'size'
            else:
                group_idx = 'amount'

            for j, method in enumerate(list(dict_df.keys())):
                df_plot = dict_df[method].groupby([group_idx])
                if len(mask_size) > len(box_amount):
                    x_mask = [m +0.25*j for m in mask_size]
                else:
                    x_mask = [m + 0.25*j for m in box_amount]
                plt.errorbar(
                    x_mask, 
                    df_plot[metric].mean(),
                    yerr=df_plot[metric].std(),
                    c=l_colors[j]
                )
                
                for amount, size in list(itertools.product(box_amount, mask_size)):
                    dict_idx = {'amount':amount, 'size':size}
                    if print_loss: 
                        print(f'{amount} | {size} | {metric} | {method} -  ' \
                            + f'Mean: {df_plot[metric].mean().loc[dict_idx[group_idx]]:.8f}, ' \
                            + f'Std: {df_plot[metric].std().loc[dict_idx[group_idx]]:7f}, ' \
                            + f'Median: {df_plot[metric].median().loc[dict_idx[group_idx]]:7f}')
            
            plt.legend(labels)
            plt.ylabel(f'{metric}')
            timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
            if len(mask_size) > len(box_amount):
                plt.title(f'Single mask')
                plt.xlabel('Mask size')
                plt.xticks(mask_size)
                plt.savefig(f'{output_path}/plots/size_{timestamp}_{metric}.png')
            else:
                plt.title(f'Multiple masks')
                plt.xlabel('Mask amount')
                plt.xticks(box_amount)
                plt.savefig(f'{output_path}/plots/amount_{timestamp}_{metric}.png')    


if __name__ == '__main__':
    args = parser.parse_args()

    predict(
        exp=args.exp,
        name=args.name,
        cfg_file=args.cfg_file,
        num_samples=args.num_samples,
        box_amount=args.box_amount,
        mask_size=args.mask_size,
        method=args.method,
        lab=args.lab,
        plot=args.plot,
        err_scale=args.err_scale,
        save=args.save,
    )
