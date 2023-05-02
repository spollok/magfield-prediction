import os
import torch
import numpy as np

from torchvision.transforms.functional import crop
from torchvision.transforms import RandomRotation
from yaml.loader import SafeLoader


def field_loader(path, factor=1, lab=False):
    with open(path, 'rb') as f:
        field = np.expand_dims(np.load(f) * factor, axis=0)

    if lab:
        field = field[:,:,:88,:88]
        rotater = RandomRotation(degrees=(-91, -89))
        field = torch.from_numpy(field.astype('float32')) * (-1)
        field = rotater(field)
        field_x = field[:,0].clone()
        field_y = field[:,1].clone()
        field[:,0] = field_y
        field[:,1] = field_x
    
    return field


# Change the values of tensor x from range [0, 1] to [-1, 1]
def normalize(x):
    return x.mul_(2).add_(-1)


def calc_div(field, dim_two):
    Hx_x = torch.gradient(field[0,0], dim=1)[0]
    Hy_y = torch.gradient(field[0,1], dim=0)[0]

    if not dim_two: 
        Hz_z = torch.gradient(field[0,2], dim=2)[0]
        div_mag = torch.stack([Hx_x, Hy_y, Hz_z], dim=0)[:,:,:,1]
    else:
        div_mag = torch.stack([Hx_x, Hy_y], dim=0)

    return torch.mean(torch.abs(div_mag.sum(dim=0)))


def calc_curl(field, dim_two):
    Hx_y = torch.gradient(field[0,0], dim=0)[0]
    Hy_x = torch.gradient(field[0,1], dim=1)[0]

    if not dim_two:
        Hx_z = torch.gradient(field[0,0], dim=2)[0]
        Hy_z = torch.gradient(field[0,1], dim=2)[0]
        Hz_x = torch.gradient(field[0,2], dim=1)[0]
        Hz_y = torch.gradient(field[0,2], dim=0)[0]
        curl_vec = torch.stack([Hz_y - Hy_z, Hx_z - Hz_x, Hy_x - Hx_y], dim=0)[:,:,:,1]
        curl_mag = curl_vec.square().sum(dim=0)
    else:
        curl_mag = (Hy_x - Hx_y).square()

    return torch.mean(curl_mag)


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']

    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


def random_bbox(config, distributed=False, rng=None):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img
        seed: When set, then bbox creation is deterministic

    Returns:
        tuple: (top, left, height, width)
        List of tuples for each sample in batch

    """
    if rng is None: rng = np.random.default_rng()
    _, img_height, img_width = config['image_shape']
    h, w = config['mask_shape']
    margin_height, margin_width = config['margin']
    maxt = img_height - margin_height - h
    maxl = img_width - margin_width - w
    bbox_list = []
    if config['mask_batch_same']:
        # List for multiple boxes in one sample
        bbox_list.append([])
        for i in range(config['box_amount']):
            if distributed:
                num_dim = int(np.sqrt(config['box_amount']))
                pos_h = np.linspace(margin_height, maxt, num_dim + 1)
                pos_w = np.linspace(margin_width, maxl, num_dim)
                idx_h = i // num_dim
                idx_w = i % num_dim
                t = pos_h[idx_h]
                l = pos_w[idx_w]
            else:
                t = rng.integers(low=margin_height, high=maxt)
                l = rng.integers(low=margin_width, high=maxl)
            bbox_list[-1].append((t, l, h, w))
        bbox_list = bbox_list * config['batch_size']
    else:
        for _ in range(config['batch_size']):
            t = rng.integers(low=margin_height, high=maxt)
            l = rng.integers(low=margin_width, high=maxl)
            bbox_list.append((t, l, h, w))

    return torch.tensor(bbox_list, dtype=torch.int64) # pylint: disable=E1102

def random_bnd(mask, perc):

    length = mask.shape[-1]
    perc_array = torch.ones([length*4])
    perc_array[0:int(len(perc_array)*(perc/100))] = 0 
    # np.random.shuffle(perc_array)
    r = torch.randperm(perc_array.shape[0])
    perc_array = perc_array[r]
    mask[:,:,0,:] = perc_array[:length]
    mask[:,:,-1,:] = perc_array[length:length*2]
    mask[:,:,:,0] = perc_array[length*2:length*3]
    mask[:,:,:,-1] = perc_array[length*3:]

    return mask

def random_bnd2(mask, perc):

    length = mask.shape[-1]
    prob = 0.1
    rng = np.random.default_rng()
    perc_array = rng.choice([0, 1], size=(4, 96), p=[prob, 1 - prob])
    mask[:,:,0,:] = perc_array[0,:]
    mask[:,:,-1,:] = perc_array[1,:]
    mask[:,:,:,0] = perc_array[2,:]
    mask[:,:,:,-1] = perc_array[3,:]

    return mask


# bboxes is the output from random_bbox (torch.tensor)
def bbox2mask(bboxes, height, width, max_delta_h, max_delta_w, bnd, outpaint=False):
    batch_size = bboxes.size(0)
    # Adding boundary condition
    if bnd is not None:
        mask_size = (batch_size, 1, int(bboxes[0,0][2] + 2 * bnd), int(bboxes[0,0][3] + 2 * bnd))
    else:
        mask_size = (batch_size, 1, height, width)

    if outpaint:
        mask = torch.ones(mask_size, dtype=torch.float32)
    else:
        mask = torch.zeros(mask_size, dtype=torch.float32)

    # Faster implementation for mask_batch_same is True
    for bbox in bboxes[0]:
        delta_h = np.random.randint(max_delta_h // 2 + 1)
        delta_w = np.random.randint(max_delta_w // 2 + 1)

        init_t = bbox[0] + delta_h if bnd is None else bnd
        init_l = bbox[1] + delta_w if bnd is None else bnd
        end_t = max(bbox[0] + delta_h + 1, bbox[0] + bbox[2] - delta_h) if bnd is None else mask_size[2] - bnd
        end_l = max(bbox[1] + delta_w + 1, bbox[1] + bbox[3] - delta_w) if bnd is None else mask_size[3] - bnd

        mask[
                :,:,
                init_t:end_t,
                init_l:end_l
            ] = 0. if outpaint else 1.

    return mask


def local_patch(x, bboxes, s, outpaint, mode):
    assert len(x.size()) == 4
    if mode == None:
        mode = 'inpaint'
    _, _, h, w = bboxes[0][0]
    patch_dict = {
        'full': ((x.shape[2] - h) + (x.shape[3] - w), x.shape[3]),
        'full_eco': (x.shape[2] - h, x.shape[3] + w),
        'small': (4 * s, 2 * s + w),
        'small_sep': (s, 2 * s + w),
        'one': (2 * s + h, 2 * s + w),
        'extend': (2 * s + h, 2 * s + w),
        'inpaint': (h, w),
    }
    h_patch, w_patch = patch_dict[mode]

    patches = []
    bbox_list = bboxes[0]
    
    for bbox in bbox_list:
        t, l, h, w = bbox
        if outpaint:
            # Connecting patches with overlap ==> img_shape[1] == img_shape[2]
            if mode == 'full':
                t_patch = crop(x, 0, 0, t, x.shape[3])
                b_patch = crop(x, t + h, 0, x.shape[2] - (t + h), x.shape[3])
                l_patch = crop(x, 0, 0, x.shape[2], l).transpose(2,3)
                r_patch = crop(x, 0, l + w, x.shape[2], x.shape[3] - (l + w)).transpose(2,3)
                patch = [torch.cat((t_patch, b_patch, l_patch, r_patch), dim=2)]
            
            # Connecting patches over img_shape[1] - mask_shape[0] ==> h == w & img_shape[1] == img_shape[2]
            elif mode == 'full_eco':
                t_patch = crop(x, 0, 0, t, x.shape[3])
                b_patch = crop(x, t + h, 0, x.shape[2] - (t + h), x.shape[3])
                l_patch = crop(x, t, 0, h, l).transpose(2,3)
                r_patch = crop(x, t, l + w, h, x.shape[3] - (l + w)).transpose(2,3)
                patch = [torch.cat((t_patch, b_patch, l_patch, r_patch), dim=2)]

            # Small local patches around given box
            if 'small' in mode:
                t_patch = crop(x, max(0, t - s), torch.clamp(l - s, 0, x.shape[3] - (w + 2 * s)), s, w + 2 * s)
                b_patch = crop(x, min(t + h, x.shape[2] - s), torch.clamp(l - s, 0, x.shape[3] - (w + 2 * s)), s, w + 2 * s)
                l_patch = crop(x, torch.clamp(t - s, 0, x.shape[2] - (h + 2 * s)), max(0, l - s), h + 2 * s, s).transpose(2,3)
                r_patch = crop(x, torch.clamp(t - s, 0, x.shape[2] - (h + 2 * s)), min(l + w, x.shape[3] - s), h + 2 * s, s).transpose(2,3)

                if mode == 'small':
                    patch = [torch.cat((t_patch, b_patch, l_patch, r_patch), dim=2)]
                elif mode == 'small_sep':
                    patch = [t_patch, b_patch, l_patch, r_patch]

            # Patch around each bbox
            elif mode == 'one':
                patch = [crop(x, torch.clamp(t - s, 0, x.shape[2] - (h + 2 * s)), torch.clamp(l - s, 0, x.shape[3] - (w + 2 * s)), h + 2 * s, w + 2 * s)]
        
        else:
            if mode == 'extend':
                patch = [crop(x, torch.clamp(t - s, 0, x.shape[2] - (h + 2 * s)), torch.clamp(l - s, 0, x.shape[3] - (w + 2 * s)), h + 2 * s, w + 2 * s)]
            else:
                patch = [crop(x, t, l, h, w)]
        
        for pat in patch:
            patches.append(pat)

    return torch.stack(patches, dim=0).reshape(-1, x.shape[1], h_patch, w_patch)


def mask_image(x, bboxes, config, bnd=None, perc = 100):
    _, height, width = config['image_shape']
    max_delta_h, max_delta_w = config['max_delta_shape']
    mask = bbox2mask(bboxes, height, width, max_delta_h, max_delta_w, bnd, config['outpaint'])
    mask = random_bnd(mask, perc=perc)
    if type(x) != np.ndarray:
        if x.is_cuda:
            mask = mask.cuda()
    else:
        mask = mask.cpu().data.numpy()
    (t,l,h,w) = bboxes[0,0]

    if bnd is None:
        original = x
        result = x * (1. - mask)
    else:
        original = x[
            :,:,
            t - bnd:t + h + bnd,
            l - bnd:l + w + bnd
        ]
        result = original * (1. - mask)

    return result, mask, original


def patch_mask(config):
    height, width = config['mask_shape']
    if config['outpaint']:
        # Patch size for different modes
        if config['mode'] == 'small':
            patch_h = 4 * config['box_patch']
            patch_w = 2 * config['box_patch'] + config['mask_shape'][1]
        elif config['mode'] == 'small_sep':
            patch_h = config['box_patch']
            patch_w = 2 * config['box_patch'] + config['mask_shape'][1]
        elif config['mode'] == 'one':
            patch_h = 2 * config['box_patch'] + config['mask_shape'][0]
            patch_w = 2 * config['box_patch'] + config['mask_shape'][1]
        shape = [1, 1, patch_h, patch_w]
    else:
        if config['mode'] == 'extend':
            patch_h = 2 * config['box_patch'] + config['mask_shape'][0]
            patch_w = 2 * config['box_patch'] + config['mask_shape'][1]
        else:
            patch_h = height
            patch_w = width
        shape = [1, 1, patch_h, patch_w]
        
    mask_values = torch.tensor(np.ones(shape), dtype=torch.float32)
    if config['cuda']:
        mask_values = mask_values.cuda()

    return mask_values


def reduce_mean(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.mean(x, dim=i, keepdim=keepdim)
    return x


def reduce_std(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.std(x, dim=i, keepdim=keepdim)
    return x


def reduce_sum(x, axis=None, keepdim=False):
    if not axis:
        axis = range(len(x.shape))
    for i in sorted(axis, reverse=True):
        x = torch.sum(x, dim=i, keepdim=keepdim)
    return x


def get_config(config):
    import yaml
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=SafeLoader)


def get_model_list(dirname, key, iteration=0, best=False):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()

    if best:
        for model_name in gen_models:
            if 'best' in model_name:
                return model_name
        raise ValueError('No best models in this experiment')

    if iteration == 0:
        last_model_name = gen_models[-1]
    else:        
        for model_name in gen_models:
            if '{:0>8d}'.format(iteration) in model_name:
                return model_name
        raise ValueError('Not found models with this iteration')
    
    return last_model_name
