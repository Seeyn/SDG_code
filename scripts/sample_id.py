import argparse
import os
import time

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import torchvision
import torch.nn.functional as F
import torch 

from sdg.parser import create_argparser
from sdg.logging import init_logging, make_logging_dir
from sdg.distributed import master_only_print as print
from sdg.distributed import is_master, init_dist, get_world_size
from sdg.gpu_affinity import set_affinity
from sdg.logging import init_logging, make_logging_dir
from sdg.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from sdg.clip_guidance import CLIP_gd
from sdg.image_datasets import load_ref_data
from sdg.misc import set_random_seed
from sdg.guidance import image_loss, text_loss
from sdg.image_datasets import _list_image_files_recursively
from torchvision import utils
import math
import clip
from sdg.face_id import ResNetArcFace

def main():

    gt_sr_mask = {}
    name_ = None

    time0 = time.time()
    args = create_argparser().parse_args()
    set_affinity(args.local_rank)
    if args.randomized_seed:
        args.seed = random.randint(0, 10000)
    set_random_seed(args.seed, by_rank=True)
    if not args.single_gpu:
        init_dist(args.local_rank)

    tb_log = None
    args.logdir = init_logging(args.exp_name, root_dir='results', timestamp=False)
    if is_master():
        tb_log = make_logging_dir(args.logdir, no_tb=True)

    print("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    model.to('cuda')
    model.eval()
    clip_ft = CLIP_gd(args)
    clip_ft.load_state_dict(th.load(args.clip_path, map_location='cpu'))
    clip_ft.eval()
    clip_ft = clip_ft.cuda()



    face_p = ResNetArcFace('IRBlock', [2,2,2,2], use_se=False)
    weights = th.load('/home/tangb_lab/cse30013027/lzl/SDG_code/logs/tune_id/model000500.pt')
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v

    face_p.load_state_dict(weights_dict)
    face_p.eval()
    face_p.cuda()

    face_p_ori = ResNetArcFace('IRBlock', [2,2,2,2], use_se=False)
    weights = th.load('./arcface_resnet18.pth')
    weights_dict = {}
    for k, v in weights.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
    face_p_ori.load_state_dict(weights_dict)
    face_p_ori.eval()
    face_p_ori.cuda()

    cri  = th.nn.MSELoss(reduction='mean')
    # define image list
    if args.image_weight == 0:
        imgs = [None]
    else:
        imgs = _list_image_files_recursively(args.data_dir)
        imgs = sorted(imgs)
    
    def gray_resize_for_identity(out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def cond_fn_sdg(x, t, y, **kwargs):
        assert y is not None
        #with th.no_grad():
            #target_img_noised = diffusion.q_sample(kwargs['ref_img'], t, tscale1000=True)
            #target_img_features = clip_ft.encode_image_list(target_img_noised, t)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            #image_features = clip_ft.encode_image_list(x_in, t)
            #print(x_in.shape)
            out = face_p(gray_resize_for_identity(x_in))
            gt_sr_mask[name_].append(out)
            #print(y.shape,out.shape)
            loss_mask = cri(out,y)

            #loss_img = image_loss(image_features, target_img_features, args)
            print(loss_mask.sum()) 
            total_guidance =  -1000*loss_mask #+  loss_img.mean() * args.image_weight

            return th.autograd.grad(total_guidance.sum(), x_in)[0]


    print("creating samples...")
    count = 0
    for img_cnt in range(len(imgs)):
        if imgs[img_cnt] is not None:
            print("loading data...")
            model_kwargs = load_ref_data(args, imgs[img_cnt])
        else:
            model_kwargs = {}
        model_kwargs['y'] = face_p_ori(gray_resize_for_identity(model_kwargs["ref_img"].cuda()))
        model_kwargs = {k: v.to('cuda') for k, v in model_kwargs.items()}
        name_ =os.path.basename(imgs[img_cnt]).split('.')[0]
        gt_sr_mask[name_] = []
        gt_sr_mask[name_].append(model_kwargs['y'])

        if args.image_weight == 0 and args.text_weight == 0:
            cond_fn = None
        else:
            cond_fn = cond_fn_sdg
        with th.cuda.amp.autocast(True):
                sample = diffusion.p_sample_loop(
                    model,
                    (args.batch_size, 3, args.image_size, args.image_size),
                    noise=None,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=model_kwargs,
                    cond_fn=cond_fn,
                    device='cuda',
                )

        for i in range(args.batch_size):

            out_folder = '%05d_%05d_%s_%s' % (img_cnt,0 , os.path.basename(imgs[img_cnt]).split('.')[0], 'parse')

            out_path = os.path.join(args.logdir, out_folder,
                                        f"{str(count * args.batch_size + i).zfill(5)}.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            utils.save_image(
                    sample[i].unsqueeze(0),
                    out_path,
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )

        count += 1
        print(f"created {count * args.batch_size} samples")
        print(time.time() - time0)
        np.save('id_xt_wovar',gt_sr_mask)
    print("sampling complete")


if __name__ == "__main__":
    main()
