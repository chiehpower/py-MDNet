import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config import gen_config

opts = yaml.safe_load(open('tracking/options.yaml','r'))

# Extract pos/neg features
def forward_samples(model, image, samples, out_layer='conv3', Show_image=False):
    model.eval()
    # image.size = original size
    extractor = RegionExtractor(image, samples, opts, Show_image)
    print("\n Forward samples \n")
    for i, regions in enumerate(extractor):
        # regions.size = 256,3,107,107
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
            # feat = 512*3*3
            # feat.size = 256,4608
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    # image.size = original size
    return feats

# train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()
    # 每個mini-batch，圖片上隨機採32個正樣本 和96個負樣本
    batch_pos = opts['batch_pos'] # 32 
    batch_neg = opts['batch_neg'] # 96
    batch_test = opts['batch_test'] # 256
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg) # 1024

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0
    # maxiter = 50
    for i in range(maxiter):
        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        ## pos_score.shape 32 neg_score.shape 96
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()

## result, result_bb, fps = result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)
def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    ## img_list : absolute path of our datasets 
    # Init bbox
    print("a. Init bbox")
    target_bbox = np.array(init_bbox)
    result = np.zeros((len(img_list), 4))
    result_bb = np.zeros((len(img_list), 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(len(img_list))
        overlap[0] = 1

    # Init model
    print("b. Init model")
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()

    # Init criterion and optimizer 
    print("c. Init criterion and optimizer")
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image # 讀取第一幀
    print("d. Load first image")
    image = Image.open(img_list[0]).convert('RGB')

    # Draw pos/neg samples
    print("e. Draw pos/neg samples")
    # 用第一幀來畫正負樣本
    # training examples sampling
        ## trans_pos: 0.1 
        ## scale_pos: 1.3
        ## n_pos_init: 500 (最大集合)
        ## overlap_pos_init: [0.7, 1]

    print("Training examples sampling ...")
    # pos_examples.shape = (500,4)
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])
    # Multi-domain model samples
        ## trans_neg_init: 1
        ## scale_neg_init: 1.6
        ## n_neg_init: 5000 (最大集合)
        ## overlap_neg_init: [0, 0.5]

    print("Multi-domain model samples ...")
    # neg_examples.shape = (5000,4)
    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # 已經先用第一章圖片製造出500個正樣本和5000個負樣本。

    # Extract pos/neg features
    print("Extract pos/neg features...")
    # Forward samples (2 times)
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([244, 3, 107, 107])
    # feat.size : torch.Size([244, 4608])

    pos_feats = forward_samples(model, image, pos_examples, Show_image=False)

    # Batch size is 256 and neg_sample size is 5000
    # 5000/256 = 19.53125 Hence, 256*19 = 4864 
    # 4864 + 136 = 5000 > that is why total 20 times.

    # Forward samples (20 times)
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([256, 3, 107, 107])
    # feat.size : torch.Size([256, 4608])
    # regions.size : torch.Size([136, 3, 107, 107])
    # feat.size : torch.Size([136, 4608])
    neg_feats = forward_samples(model, image, neg_examples, Show_image=False)

    # Initial training
    print("f. Initial training")
    ## def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4')
    ## maxiter_init: 50
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor
    ## trans_bbreg: 0.3
    ## scale_bbreg: 1.6
    ## aspect_bbreg: 1.1
    ## n_bbreg: 1000
    ## overlap_bbreg: [0.6, 1]
    print("Train bbox regressor...")
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples, Show_image=False)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    print("g. Init sample generators for update")
    ## trans: 0.6
    ## scale: 1.05
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    ## trans_pos: 0.1
    ## scale_pos: 1.3
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    ## trans_neg: 2
    ## scale_neg: 1.3
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    print("h. Init pos/neg features for update")
    ## overlap_neg_init: [0, 0.5]
    ## n_neg_update: 200
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples, Show_image=False)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Display
    print("i. Display")
    savefig = savefig_dir != ''
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi)

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto')

        if gt is not None: 
            # Green
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)
        # red
        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            print("Save the 0000.jpg picture")
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)


    # Main loop
    print("j. Main Loop")
    for i in range(1, len(img_list)):
        tic = time.time()
        # Load image
        image = Image.open(img_list[i]).convert('RGB')

        # Estimate target bbox
        # 生成256個候選匡
        # n_samples : 256
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6', Show_image=False)
        # sample_scores.shape = torch.Size([256, 2])
        
        # top_scores tensor([15.6350, 15.5444, 15.3375, 11.5216,  9.8178], device='cuda:0')
        # top_idx tensor([172, 164,  63, 187, 115], device='cuda:0')
        top_scores, top_idx = sample_scores[:, 1].topk(5) # Pick the largest five and include his position
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats = forward_samples(model, image, bbreg_samples, Show_image=False)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples, Show_image=False)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples, Show_image=False)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image)

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '{:04d}.jpg'.format(i)), dpi=dpi)

        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list), target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i, len(img_list), overlap[i], target_score, spf))

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = len(img_list) / spf_total
    return result, result_bb, fps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    assert args.seq != '' or args.json != ''

    np.random.seed(0)
    torch.manual_seed(0)

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
    result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
