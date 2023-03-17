import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from kornia.filters import SpatialGradient

from model.networks import Generator, GlobalDis
from utils.tools import get_model_list
from utils.logger import get_logger

logger = get_logger()


class Trainer(nn.Module):
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.box_patch = self.config['box_patch']
        self.outpaint = self.config['outpaint']
        self.mode = self.config['mode']
        self.coarse_G = self.config['coarse_G']

        self.netG = Generator(
            self.config['netG'],
            self.config['coarse_G'],
            self.use_cuda,
            self.device_ids
        )
        self.globalD = GlobalDis(
            self.config['netD'],
            self.config['image_shape'],
            self.config['mask_shape'],
            self.config['boundary'],
            self.use_cuda,
            self.device_ids
        )

        self.optimizer_g = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.config['lr'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        self.optimizer_d = torch.optim.Adam(
            list(self.globalD.parameters()),
            lr=config['lr'],
            betas=(self.config['beta1'], self.config['beta2'])
        )
        lambda0 = lambda epoch: 0.97 ** (epoch * 0.0001)
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer_d, lr_lambda=lambda0)

        if self.use_cuda:
            self.netG.to(self.device_ids[0])
            self.globalD.to(self.device_ids[0])

    def forward(self, x, bboxes, mask, gt, gt_top, gt_bottom, compute_loss_g=False):
        self.train()
        l1_loss = nn.L1Loss()
        losses = {}

        x1, x2, phi = self.netG(x, mask)
        if self.outpaint:
            if x1 is not None: x1_eval = x1
            x2_eval = x2
        else:
            if x1 is not None: x1_eval = x1 * mask + x * (1. - mask)
            x2_eval = x2 * mask + x * (1. - mask)
        
        # Groundtruth in terms of scalar potential
        gt_phi = SpatialGradient(gt)
        # D part
        # wgan d loss
        global_real_pred, global_fake_pred = self.dis_forward(self.globalD, gt_phi, phi.detach())
        losses['wgan_d'] = torch.mean(global_fake_pred - global_real_pred) * self.config['global_wgan_loss_alpha']
        # gradients penalty loss
        global_penalty = self.calc_gradient_penalty(self.globalD, gt_phi, phi.detach())
        losses['wgan_gp'] = global_penalty

        # G part
        if compute_loss_g:
            losses['l1'] = l1_loss(phi, gt_phi)
            losses['ae'] = l1_loss(phi * (1. - mask), gt_phi * (1. - mask))
            if x1 is not None:
                losses['l1'] += l1_loss(x1_eval, gt) * self.config['coarse_l1_alpha']
                losses['ae'] += l1_loss(x1 * (1. - mask), gt * (1. - mask)) * self.config['coarse_l1_alpha']

            # Shape: bs x comp x res_h (y) x res_w (x) x z
            if self.config['netG']['input_dim'] == 3:
                field = torch.cat([gt_top.unsqueeze(-1), x2_eval.unsqueeze(-1), gt_bottom.unsqueeze(-1)], dim=-1)
            else:
                field = x2_eval
            
            # Div loss
            if self.config['div_loss']:
                Hx_x = torch.gradient(field[:,0], dim=2)[0]
                Hy_y = torch.gradient(field[:,1], dim=1)[0]
                if self.config['netG']['input_dim'] == 3:
                    Hz_z = torch.gradient(field[:,2], dim=3)[0]
                    # Taking gradients of center layer only
                    div_mag = torch.stack([Hx_x, Hy_y, Hz_z], dim=1)[:,:,:,:,1]
                else:                    
                    div_mag = torch.stack([Hx_x, Hy_y], dim=1)
                losses['div'] = torch.mean(torch.abs(div_mag.sum(dim=1)))

            # Curl
            if self.config['curl_loss']:
                Hx_y = torch.gradient(field[:,0], dim=1)[0]
                Hy_x = torch.gradient(field[:,1], dim=2)[0]
                if self.config['netG']['input_dim'] == 3:
                    Hx_z = torch.gradient(field[:,0], dim=3)[0]
                    Hy_z = torch.gradient(field[:,1], dim=3)[0]
                    Hz_x = torch.gradient(field[:,2], dim=2)[0]
                    Hz_y = torch.gradient(field[:,2], dim=1)[0]
                    # Taking gradients of center layer only
                    curl_vec = torch.stack([Hz_y-Hy_z, Hx_z-Hz_x, Hy_x-Hx_y], dim=1)[:,:,:,:,1]
                    curl_mag = curl_vec.square().sum(dim=1)
                else:
                    curl_mag = (Hy_x - Hx_y).square()
                losses['curl'] = torch.mean(curl_mag)
            
            # wgan g loss
            global_real_pred, global_fake_pred = self.dis_forward(self.globalD, gt, x2_eval)
            losses['wgan_g'] = -torch.mean(global_fake_pred) * self.config['global_wgan_loss_alpha']

        return losses, x2_eval, x2, phi

    def dis_forward(self, netD, ground_truth, x_inpaint):
        assert ground_truth.size() == x_inpaint.size()
        batch_size = ground_truth.size(0)
        batch_data = torch.cat([ground_truth, x_inpaint], dim=0)
        batch_output = netD(batch_data)
        real_pred, fake_pred = torch.split(batch_output, batch_size, dim=0)

        return real_pred, fake_pred

    # Calculate gradient penalty
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.use_cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates = interpolates.requires_grad_().clone()

        disc_interpolates = netD(interpolates)
        grad_outputs = torch.ones(disc_interpolates.size())

        if self.use_cuda:
            grad_outputs = grad_outputs.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=grad_outputs, create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def inference(self, x, mask):
        self.eval()
        _, x2 = self.netG(x, mask)
        x2_eval = x2 if self.outpaint else x2 * mask + x * (1. - mask)

        return x2_eval

    def save_model(self, checkpoint_dir, iteration, best=False):
        # Save generators, discriminators, and optimizers
        if best:
            gen_name = os.path.join(checkpoint_dir, 'gen_best.pt')
        else:
            gen_name = os.path.join(checkpoint_dir, 'gen_%08d.pt' % iteration)
            dis_name = os.path.join(checkpoint_dir, 'dis_%08d.pt' % iteration)
            opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')
            torch.save({'globalD': self.globalD.state_dict()}, dis_name)
            torch.save({'gen': self.optimizer_g.state_dict(),
                        'dis': self.optimizer_d.state_dict()}, opt_name)

        torch.save(self.netG.state_dict(), gen_name)

    def resume(self, checkpoint_dir, iteration=0, test=False):
        # Load generators
        try:
            last_model_name = get_model_list(checkpoint_dir, "gen", iteration=iteration)
            self.netG.load_state_dict(torch.load(last_model_name))
            iteration = int(last_model_name[-11:-3])

            if not test:
                # Load discriminators
                last_model_name = get_model_list(checkpoint_dir, "dis", iteration=iteration)
                state_dict = torch.load(last_model_name)
                self.globalD.load_state_dict(state_dict['globalD'])
                # Load optimizers
                state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
                self.optimizer_d.load_state_dict(state_dict['dis'])
                self.optimizer_g.load_state_dict(state_dict['gen'])
        except:
            iteration = 1

        #("Resume from {} at iteration {}".format(checkpoint_dir, iteration))
        logger.info("Resume from {} at iteration {}".format(checkpoint_dir, iteration))

        return iteration
