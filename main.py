import os
import shutil
import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import push_greedy
import argparse
import re
import numpy as np
import random
import wandb

from helpers import makedir
import model
#import push 
#import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import CosineLRScheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma
from as_dataset import get_datasets
from as_dataloader import make_loader
from omegaconf import OmegaConf


def main(cfg):

    wandb.init(
        config=OmegaConf.to_object(cfg),
        **cfg.get('wandb', {}),
    )

    img_size = cfg.architecture.img_size
    model_dir = cfg.train.model_dir
    makedir(model_dir, exist_ok=True)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir, exist_ok=True)

    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    
    num_train_epochs= cfg.train.num_joint_epochs + cfg.train.num_warm_epochs
    push_epochs= [i for i in range(num_train_epochs) if i % 10 == 0]

    # def set_seed(seed):
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    #     np.random.seed(seed)  # Numpy module.
    #     #random.seed(seed)  # Python random module.
    #     torch.manual_seed(seed)
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

    # seed = np.random.randint(10, 10000, size=1)[0]
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    train_dataset, val_dataset = get_datasets(cfg=cfg.data)
    train_loader = make_loader(dataset=train_dataset, cfg=cfg.data, shuffle=False, weighted=True, pin_memory=True, drop_last=True)
    val_loader = make_loader(dataset=val_dataset, cfg=cfg.data, shuffle=False, weighted=False, pin_memory=False)
    train_push_dataset = get_datasets(cfg=cfg.data, augment=[], data_type='push', normalize=False)
    train_push_loader = make_loader(dataset=train_push_dataset, cfg=cfg.data, shuffle=False, weighted=True, pin_memory=False)

    # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('val set size: {0}'.format(len(val_loader.dataset)))
    log('batch size: {0}'.format(cfg.data.batch_size))

    # construct the model
    arch = cfg.architecture.base_architecture
    ppnet = model.construct_PPNet(base_architecture=cfg.architecture.base_architecture,
                                pretrained=True, img_size=cfg.architecture.img_size,
                                prototype_shape=tuple(cfg.architecture[arch].prototype_shape),
                                radius = cfg.architecture.radius,
                                num_classes=cfg.train.num_classes,
                                prototype_activation_function=cfg.train.prototype_activation_function,
                                sig_temp = cfg.train.sig_temp,
                                add_on_layers_type=cfg.train.add_on_layers_type)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(str(device))
    ppnet = ppnet.to(device)
    model_ema = None
    class_specific = True

    joint_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': cfg.train.joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    #{'params': ppnet.patch_select, 'lr': joint_optimizer_lrs['patch_select']},
    {'params': ppnet.prototype_vectors, 'lr': cfg.train.joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.AdamW(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=cfg.train.joint_lr_step_size, gamma=0.1)

    # to train the slots 
    joint_optimizer_specs_stage2 =[{'params': ppnet.patch_select, 'lr': cfg.train.stage_2_lrs['patch_select']}]

    joint_optimizer2 = torch.optim.AdamW(joint_optimizer_specs_stage2)
    joint_lr_scheduler2 = torch.optim.lr_scheduler.StepLR(joint_optimizer2, step_size=cfg.train.joint_lr_step_size, gamma=0.1)

    warm_optimizer_specs = \
    [{'params': ppnet.features.parameters(), 'lr': cfg.train.warm_optimizer_lrs['features'], 'weight_decay': 1e-3},
    {'params': ppnet.prototype_vectors, 'lr': cfg.train.warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.AdamW(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': cfg.train.last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)

    # weighting of different training losses
    coefs = cfg.train.coefs

    # number of training epochs, number of warm epochs, push start epoch, push epochs
    # train the model
    log('start training')
    log(f'weight coefs are: {coefs}')
    import copy

    # check_base = False 
    # if check_base:
    #     # check the performance of base-arch 
    #     n_examples = 0
    #     n_correct = 0
    #     for i, (image, label) in enumerate(val_loader):
    #         image = image.to(device)
    #         label = label.to(device)
    #         out = ppnet.features(image)
    #         _, predicted = torch.max(out.data, 1)
    #         n_examples += label.size(0)
    #         n_correct += (predicted == label).sum().item()
    #     log('base-arch acc: \t{0}'.format(n_correct / n_examples * 100))

    #slots_epoch = num_warm_epochs +5 
    # only_push = False 
    # if not only_push:
    # not ready for push yet
    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < cfg.train.num_warm_epochs:
            tnt.warm_only(model=ppnet, log=log)
            _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = cfg.train.k, sum_cls = cfg.train.sum_cls)
        else:
            tnt.joint(model=ppnet, log=log)
            # to train the model with no slots 
            _ , train_loss= tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer,
                        class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = cfg.train.k, sum_cls = cfg.train.sum_cls)
            joint_lr_scheduler.step()

        accu, tst_loss = tnt.test(model=ppnet, dataloader=val_loader,
                        class_specific=class_specific, log=log, clst_k=cfg.train.k,sum_cls = cfg.train.sum_cls)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    target_accu=0.75, log=log)
        # version 1, learn slots before push 
    from settings import coefs_slots
    coh_weight = cfg.train.coefs_slots['coh']
    coefs['coh']  = coh_weight
    log(f'Coefs for slots training: {coefs}')
    for epoch in range(cfg.train.slots_train_epoch):
        tnt.joint(model=ppnet, log=log)
        log('epoch: \t{0}'.format(epoch))
        _ , train_loss= tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer2,
                        class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = cfg.train.k, sum_cls = cfg.train.sum_cls)
        joint_lr_scheduler2.step()
        accu, tst_loss = tnt.test(model=ppnet, dataloader=val_loader,
                        class_specific=class_specific, log=log, clst_k=cfg.train.k, sum_cls = cfg.train.sum_cls)

        wandb.log({
            "epoch": epoch
            "train_loss": train_loss,
            "val_loss": tst_loss,
            "accuracy": accu
        }    
        )
        
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'slots', accu=accu,
                                    target_accu=0.75, log=log)

    push_greedy.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            pnet = ppnet, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
    accu, tst_loss = tnt.test(model=ppnet, dataloader=val_loader,
                    class_specific=class_specific, log=log, clst_k = cfg.train.k, sum_cls = cfg.train.sum_cls)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                target_accu=0.0, log=log)

    for epoch in range(15):
        tnt.last_only(model=ppnet, log=log)
        log('iteration: \t{0}'.format(epoch))
        _, train_loss = tnt.train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, ema = model_ema, clst_k = cfg.train.k, sum_cls = cfg.train.sum_cls)
        print('Accuracy is:')
        accu, tst_loss = tnt.test(model=ppnet, dataloader=val_loader,
                            class_specific=class_specific, log=log, clst_k = cfg.train.k,sum_cls = cfg.train.sum_cls)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'finetuned', accu=accu,
                                target_accu=0.70, log=log)
    logclose()

def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help='Path to config file')
    parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    cfg = load_config(args.config)
    main(cfg)