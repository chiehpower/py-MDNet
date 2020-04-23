import os, sys
import pickle
import yaml
import time
import argparse
import numpy as np

import torch

# FIXME:
import torch.onnx
import onnxruntime
import onnx

sys.path.insert(0,'.')
from data_prov import RegionDataset
from modules.model import MDNet, set_optimizer, BCELoss, Precision


def train_mdnet(opts):

    # Init dataset
    with open(opts['data_path'], 'rb') as fp:
        data = pickle.load(fp)
    K = len(data)
    # K is how many classes
    dataset = [None] * K
    for k, seq in enumerate(data.values()):
        dataset[k] = RegionDataset(seq['images'], seq['gt'], opts)
    # Init model
    model = MDNet(opts['init_model_path'], K)
    if opts['use_gpu']:
        model = model.cuda()
    model.set_learnable_params(opts['ft_layers'])

    # Init criterion and optimizer
    criterion = BCELoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, opts['lr'], opts['lr_mult'])

    # Main trainig loop
    for i in range(opts['n_cycles']):
        print('==== Start Cycle {:d}/{:d} ===='.format(i + 1, opts['n_cycles']))

        if i in opts.get('lr_decay', []):
            print('decay learning rate')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= opts.get('gamma', 0.1)
                # FIXME:
                print("param_group['lr']",param_group['lr'])
        # Training
        model.train()
        prec = np.zeros(K)
        k_list = np.random.permutation(K) # random data order
        # For example : k_list [ 5  1 13  0 12  9  8 11 14  6  7 10  4 15  2  3]
        for j, k in enumerate(k_list):
            tic = time.time()
            # training
            pos_regions, neg_regions = dataset[k].next()
            # pos_regions.shape = torch.Size([32, 3, 107, 107])
            # neg_regions.shape = torch.Size([96, 3, 107, 107])
            if opts['use_gpu']:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()
            pos_score = model(pos_regions, k) # [32,2]
            neg_score = model(neg_regions, k) # [96,2]

            loss = criterion(pos_score, neg_score)

            batch_accum = opts.get('batch_accum', 1)
            if j % batch_accum == 0:
                model.zero_grad()
            loss.backward()
            if j % batch_accum == batch_accum - 1 or j == len(k_list) - 1:
                if 'grad_clip' in opts:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
                optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            # Domain : classes
            print('Cycle {:2d}/{:2d}, Iter {:2d}/{:2d} (Domain {:2d}), Loss {:.3f}, Precision {:.3f}, Time {:.3f}'
                    .format(i, opts['n_cycles'], j, len(k_list), k, loss.item(), prec[k], toc))

        print('Mean Precision: {:.3f}'.format(prec.mean()))
        print('Save model to {:s}'.format(opts['model_path']))
        if opts['use_gpu']:
            model = model.cpu()
        states = {'shared_layers': model.layers.state_dict()}
        torch.save(states, opts['model_path'])
        if opts['use_gpu']:
            model = model.cuda()
    # x = torch.randn(len(k_list), 3, 107, 107, requires_grad=True).cuda()  # Input
    x = torch.randn(32, 3, 107, 107, requires_grad=True).cuda()  # Input
    # torch.Size([1, 3, 107, 107]) torch.Size([32, 3, 107, 107]) torch.Size([32, 2])

    print('pos_regions: ', x.shape, pos_regions.shape, pos_score.shape) 
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "/home/chieh/github/py-MDNet/models/onnx/10/MDNet_nolrn.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    # do_constant_folding=True,  # whether to execute constant folding for optimization
                    # input_names = ['input'],   # the model's input names
                    # output_names = ['output']
                    # verbose=True
                    ) # the model's output names
                    # dynamic_axes={'input' : {0 : 'len(k_list)'},    # variable lenght axes
                    #                 'output' : {0 : 'len(k_list)'}})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='imagenet', help='training dataset {vot, imagenet}')
    args = parser.parse_args()

    opts = yaml.safe_load(open('pretrain/options_{}.yaml'.format(args.dataset), 'r'))
    train_mdnet(opts)
