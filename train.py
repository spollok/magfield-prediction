import os
import random
import time
import shutil
from pathlib import PurePath, Path
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from trainer import Trainer
from utils.dataset import Dataset
from utils.tools import get_config, random_bbox, mask_image
from utils.logger import get_logger

import matplotlib.pyplot as plt
import matplotlib.colors as col

import wandb

#To be able to run from command prompt (changin parameters from config withour actually doing so=)
parser = ArgumentParser()
parser.add_argument(
    '--config', type=str,
    default=Path(__file__).parent.resolve() / 'configs' / 'config.yaml',
    help="training configuration"
)
parser.add_argument('--seed', type=int, help='manual seed')


def main():
    args = parser.parse_args()
    config = get_config(args.config)

    #To work with WandB
    run = wandb.init(
        project="wgan-gp_bnd1", 
        entity="andreathesis",
        config=config
    )

    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True

    # Configure checkpoint path
    mode = f"out_{config['mode']}" if config['outpaint'] else 'in'
    if config['mode'] == 'extend':
        mode += '_ext'
    if config['div_loss']:
        mode += '_div'
    if config['curl_loss']:
        mode += '_curl'
    exp_name = mode + '_' + str(config['box_amount']) + '_' \
        + str(config['mask_shape'][0]) + '_' + str(config['scale_factor'])
    if config['test']: 
        exp_name = 'test_' + exp_name
    cp_path = Path(__file__).parent.resolve() / 'checkpoints' / config['dataset_name'] / exp_name
    if not cp_path.exists():
        cp_path.mkdir(parents=True)
    # elif config['resume'] is None:
    #     print('Experiment has already been run! Terminating...')
    #     exit()
    shutil.copy(args.config, cp_path / PurePath(args.config).name)
    writer = SummaryWriter(cp_path)
    logger = get_logger(cp_path)
    best_score = 1

    logger.info("Arguments: {}".format(args))
    if args.seed is None: args.seed = random.randint(1, 10000)
    logger.info("Random seed: {}".format(args.seed))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda: torch.cuda.manual_seed_all(args.seed)

    logger.info("Configuration: {}".format(config))

    try:  # for unexpected error logging
        datapath = Path(__file__).parent.resolve() / 'data'
        logger.info(f"Training on dataset: {config['dataset_name']}")
        train_dataset = Dataset(
            datapath / config['train_data'],
            config['scale_factor'],
            image_shape=config['image_shape']
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            drop_last=True
        )
        val_dataset = Dataset(
            datapath / config['val_data'],
            scaling=config['scale_factor'],
            image_shape=config['image_shape']
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            drop_last=True
        )
        trainer = Trainer(config)
        logger.info("\n{}".format(trainer.netG))
        logger.info("\n{}".format(trainer.localD))
        logger.info("\n{}".format(trainer.globalD))
        
        #WandB
        wandb.watch(trainer, log_freq=1, log='all')

        if cuda:
            trainer = nn.parallel.DataParallel(trainer, device_ids=device_ids)
            trainer_module = trainer.module
        else:
            trainer_module = trainer

        if config['resume']:
            start_iteration = trainer_module.resume(cp_path, config['resume'])
        else:
            start_iteration = 1
        iterable_train_loader = iter(train_loader)
        iterable_val_loader = iter(val_loader)
        l1_loss = nn.L1Loss()
        
        time_count = time.time()
        
        for iteration in range(start_iteration, config['niter'] + 1):
            try:
                ground_truth = next(iterable_train_loader)
            except StopIteration:
                iterable_train_loader = iter(train_loader)
                ground_truth = next(iterable_train_loader)

            # Prepare the inputs
            gt_top = None
            gt_bottom = None
            if config['netG']['input_dim'] == 3:
                gt_top = ground_truth[:,:,:,:,0]
                gt_bottom = ground_truth[:,:,:,:,2]
                ground_truth = ground_truth[:,:,:,:,1]
            
            bboxes = random_bbox(config, seed=0)
            x, mask = mask_image(ground_truth, bboxes, config, bnd=config['boundary'])

            (t,l,h,w) = bboxes[0,0]
            ground_truth = ground_truth[:,:,t - config['boundary']:t + h + config['boundary'],l - config['boundary']:l + w + config['boundary']]
            gt_top = gt_top[:,:,t - config['boundary']:t + h + config['boundary'],l - config['boundary']:l + w + config['boundary']]
            gt_bottom = gt_bottom[:,:,t - config['boundary']:t + h + config['boundary'],l - config['boundary']:l + w + config['boundary']]

            if cuda: #in pytorch lightning this happens "in the backgorund", for pytorch you have to specify it (sends tensor to GPU)
                x = x.cuda()
                mask = mask.cuda()
                ground_truth = ground_truth.cuda()

                if gt_top is not None:
                    gt_top = gt_top.cuda()
                    gt_bottom = gt_bottom.cuda()
                
            
            ###### Forward pass ######
            compute_g_loss = iteration % config['n_critic'] == 0
            losses, inpainted_result, gen_result = trainer(x, bboxes, mask, 
                ground_truth, gt_top, gt_bottom, compute_g_loss)
            # Scalars from different devices are gathered into vectors
            for k in losses.keys():
                if not losses[k].dim() == 0: losses[k] = torch.mean(losses[k])

            ###### Backward pass ######
            # Update D
            trainer_module.optimizer_d.zero_grad()
            losses['d'] = losses['wgan_d'] + losses['wgan_gp'] * config['wgan_gp_lambda']
            losses['d'].backward()
            wandb.log({"losses D": losses['d']})

            # Update G
            if compute_g_loss:
                trainer_module.optimizer_g.zero_grad()
                losses['g'] = losses['l1'] * config['l1_loss_alpha'] \
                            + losses['ae'] * config['ae_loss_alpha'] \
                            + losses['wgan_g'] * config['gan_loss_alpha']
                if config['div_loss']: losses['g'] += losses['div'] * config['div_loss_alpha']
                if config['curl_loss']: losses['g'] += losses['curl'] * config['curl_loss_alpha']
                losses['g'].backward()
                wandb.log({"losses G": losses['g']})
                trainer_module.optimizer_g.step()

            trainer_module.optimizer_d.step()
            trainer_module.scheduler.step()

            # Log and visualization - log and update to see (change it a bit to do with WandB)
            log_losses = ['l1', 'ae', 'wgan_g', 'wgan_d', 'wgan_gp', 'g', 'd']
            
            if config['div_loss']: log_losses.extend(['div'])
            if config['curl_loss']: log_losses.extend(['curl'])

            #wandb.log({"Log losses": log_losses})

            if iteration % config['print_iter'] == 0:
                time_count = time.time() - time_count
                speed = config['print_iter'] / time_count
                speed_msg = 'speed: %.2f batches/s ' % speed
                time_count = time.time()

                message = 'Iter: [%d/%d] ' % (iteration, config['niter'])
                for k in log_losses:
                    v = losses.get(k, 0.)
                    wandb.log({str(k): v})
                    writer.add_scalar(k, v, iteration)
                    message += '%s: %.6f ' % (k, v)
                message += speed_msg
                logger.info(message)
            
            if iteration % (config['viz_iter']) == 0:
                gt = ground_truth / config['scale_factor']
                res = inpainted_result / config['scale_factor']
                err = abs(gt - res)
                mode = 'Out' if config['outpaint'] else 'In'
                plt.close('all')
                fig, axes = plt.subplots(nrows=config['netG']['input_dim'], 
                                         ncols=3, sharex=True, sharey=True)
                viz_list = [
                    ('Truth_X', gt[0,0]), (mode + 'paint_X', res[0,0]), ('Error_X', err[0,0]),
                    ('Truth_Y', gt[0,1]), (mode + 'paint_Y', res[0,1]), ('Error_Y', err[0,1])
                ]                
                if config['netG']['input_dim'] == 3:
                    viz_list.extend([
                        ('Truth_Z', gt[0,2]), (mode + 'paint_Z', res[0,2]), ('Error_Z', err[0,2])
                    ])

                for i, (title, data) in enumerate(viz_list):
                    ax = axes.flat[i]
                    ax.set_title(title)
                    if 'Error' in title:
                        _ = ax.imshow(data.cpu().data.numpy(), cmap='cividis',
                            norm=col.Normalize(vmin=0, vmax=0.005), origin="lower")
                    else:
                        im = ax.imshow(data.cpu().data.numpy(), cmap='bwr',
                            norm=col.Normalize(vmin=-0.04, vmax=0.04), origin="lower")

                fig.colorbar(im, ax=axes.ravel().tolist())
                plt.savefig(f'{cp_path}/{iteration}.png')

            # Save the model
            if iteration % config['snapshot_save_iter'] == 0:
                trainer_module.save_model(cp_path, iteration)
            
            # Validation
            if iteration % config['valid_iter'] == 0:
                with torch.no_grad():
                    val_loss = []
                    for _ in range(25):
                        try:
                            ground_truth = next(iterable_val_loader)
                        except StopIteration:
                            iterable_val_loader = iter(val_loader)
                            ground_truth = next(iterable_val_loader)
                        
                        # Extract center layer if three layers are provided
                        if len(ground_truth.shape) == 5:
                            ground_truth = ground_truth[:,:,:,:,1]
                        bboxes = random_bbox(config) # ADD SEED!!
                        x, mask = mask_image(ground_truth, bboxes, config)

                        if cuda:
                            x = x.cuda()
                            mask = mask.cuda()
                            ground_truth = ground_truth.cuda()

                        # Inference
                        _, x2 = trainer_module.netG(x, mask)
                        if config['outpaint']:
                            x2_eval = x2
                        else:
                            x2_eval = x2 * mask + x * (1. - mask)
                        val_loss.append(l1_loss(x2_eval, ground_truth))

                    wandb.log({"L1-loss (val)": val_loss})
                    val_err = sum(val_loss) / len(val_loss)

                    # Saving best model
                    if val_err < best_score:
                        logger.info(f'Saving new best model...')
                        best_score = val_err
                        trainer_module.save_model(cp_path, iteration, best=True)

                    writer.add_scalar('val_l1', val_err, iteration)
                    logger.info(f'Validation: {val_err:.6f}')

    except Exception as e:  # for unexpected error logging
        logger.error("{}".format(e))
        raise e


if __name__ == '__main__':
    main()
