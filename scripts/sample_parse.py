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
from sdg.faceparser import FaceParseNet50

def main():
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

    face_p = FaceParseNet50(num_classes=19, pretrained=False)
    face_p.load_state_dict(torch.load('../38_G.pth'))
    face_p.eval()

    # define image list
    if args.image_weight == 0:
        imgs = [None]
    else:
        imgs = _list_image_files_recursively(args.data_dir)
        imgs = sorted(imgs)

    def cond_fn_sdg(x, t, y, **kwargs):
        assert y is not None
        with th.no_grad():
            target_img_noised = diffusion.q_sample(kwargs['ref_img'], t, tscale1000=True)
            target_img_features = clip_ft.encode_image_list(target_img_noised, t)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            image_features = clip_ft.encode_image_list(x_in, t)
            print(x_in.shape)
            loss_mask = F.mse_loss(face_p(x_in),y)

            loss_img = image_loss(image_features, target_img_features, args)
           
            total_guidance =  loss_mask + loss_img * args.image_weight

            return th.autograd.grad(total_guidance.sum(), x_in)[0]


    print("creating samples...")
    count = 0
    for img_cnt in range(len(imgs)):
        if imgs[img_cnt] is not None:
            print("loading data...")
            model_kwargs = load_ref_data(args, imgs[img_cnt])
        else:
            model_kwargs = {}
            model_kwargs['y'] = face_p(imgs[img_cnt])
            model_kwargs = {k: v.to('cuda') for k, v in model_kwargs.items()}
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

    print("sampling complete")


if __name__ == "__main__":
    main()
