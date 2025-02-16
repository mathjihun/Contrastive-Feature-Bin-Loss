#!/usr/bin/env python

import numpy as np
import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from utils import post_process_depth, flip_lr, compute_errors
import argparse
import os, sys
import torch.backends.cudnn as cudnn
from dataloaders.sunrgbd import SUNRGBDDataset
from dataloaders.diode import DiodeDataset
from networks.PixelFormer import PixelFormer

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='PixelFormer PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name',                type=str,   help='model name', default='pixelformer')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='nyu')
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Preprocessing
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Eval
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')


if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()


def eval(model, dataloader_eval, post_process=False):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval)):
        with torch.no_grad():
            image = eval_sample_batched['image'].cuda()
            gt_depth = eval_sample_batched['gt']
            valid_mask = eval_sample_batched['mask'].to(torch.bool)

            '''
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue
            '''

            pred_depth = model(image)
            
            # GT-Based mean depth rescaling
            if args.dataset == 'diode':
                pred_depth = pred_depth * torch.median(gt_depth[valid_mask]) / torch.median(pred_depth[valid_mask])

                if post_process:
                    image_flipped = flip_lr(image)
                    gt_flipped = flip_lr(gt_depth)
                    mask_flipped = flip_lr(valid_mask)

                    pred_depth_flipped = model(image_flipped)
                    pred_depth_flipped = pred_depth_flipped * torch.median(gt_flipped[mask_flipped]) / torch.median(pred_depth_flipped[mask_flipped])

                    pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            elif post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(image_flipped)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)
            # pred_depth[pred_depth>8] = 8
        
            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()
            valid_mask = valid_mask.cpu().numpy().squeeze()


        # for kitti eigen
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped


        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        # 아래와 같이 valid_mask2를 사용하면 silog loss만 아주 조금 향상됨 사실상 차이 없음
        valid_mask2 = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        valid_mask = np.logical_and(valid_mask, valid_mask2)
        
        # garg_crop: outdoor, eigen_crop: indoor
        # diode, sunrgbd는 이미 되어있음
        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    eval_measures_cpu = eval_measures.cpu()
    cnt = eval_measures_cpu[9].item()
    eval_measures_cpu /= cnt
    print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                    'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                    'd3'))
    for i in range(8):
        print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
    print('{:7.3f}'.format(eval_measures_cpu[8]))
    return eval_measures_cpu


if args.dataset == 'sunrgbd':
    valid_dataset = SUNRGBDDataset(test_mode=True, base_path='/workspace/other_dataset')     # sunrgbd_val.txt location, 5050 images
elif args.dataset == 'diode':
    valid_dataset = DiodeDataset(test_mode=True, base_path='/workspace/other_dataset/diode')       # 



valid_sampler = SequentialSampler(valid_dataset)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    num_workers=4,
    sampler=valid_sampler,
    pin_memory=True,
    drop_last=False,
)


model = PixelFormer(version=args.encoder, inv_depth=False, max_depth=args.max_depth, pretrained=None)
model = torch.nn.DataParallel(model)
model = model.cuda()

if args.checkpoint_path != '':
    if os.path.isfile(args.checkpoint_path):
        print("== Loading checkpoint '{}'".format(args.checkpoint_path))
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("== Loaded checkpoint '{}'".format(args.checkpoint_path))
        del checkpoint
    else:
        print("== No checkpoint found at '{}'".format(args.checkpoint_path))

cudnn.benchmark = True

model.eval()

with torch.no_grad():
    eval_measures = eval(model, valid_loader, post_process=True)