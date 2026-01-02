import os
import shutil
import torch
import torch.utils.data
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
import dino_model
import train_and_test_debug as tnt
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
    makedir(model_dir)

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)

    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    num_train_epochs = cfg.train.num_joint_epochs + cfg.train.num_warm_epochs

    # Set random seeds for reproducibility
    seed = cfg.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load datasets
    train_dataset, val_dataset = get_datasets(cfg=cfg.data)
    train_loader = make_loader(dataset=train_dataset, cfg=cfg.data, shuffle=False, weighted=True, pin_memory=True, drop_last=True)
    val_loader = make_loader(dataset=val_dataset, cfg=cfg.data, shuffle=False, weighted=False, pin_memory=False)
    train_push_dataset = get_datasets(cfg=cfg.data, augment='none', data_type='push', normalize=False)
    train_push_loader = make_loader(dataset=train_push_dataset, cfg=cfg.data, shuffle=False, weighted=False, pin_memory=False)  # Changed: weighted=False for push

    log('training set size: {0}'.format(len(train_loader.dataset)))
    log('push set size: {0}'.format(len(train_push_loader.dataset)))
    log('val set size: {0}'.format(len(val_loader.dataset)))
    log('batch size: {0}'.format(cfg.data.batch_size))

    # Construct the model
    arch = cfg.architecture.base_architecture
    if arch == 'dinov3':
        ppnet = dino_model.construct_PPNetDINO(
        base_architecture=cfg.architecture.base_architecture,
        pretrained=True, 
        img_size=cfg.architecture.img_size,
        prototype_shape=tuple(cfg.architecture[arch].prototype_shape),
        radius=cfg.architecture.radius,
        num_classes=cfg.train.num_classes,
        prototype_activation_function=cfg.train.prototype_activation_function,
        sig_temp=cfg.train.sig_temp,
        add_on_layers_type=cfg.train.add_on_layers_type,
        layers= cfg.train.layers
    )

    else:
        ppnet = model.construct_PPNet(
            base_architecture=cfg.architecture.base_architecture,
            pretrained=True, 
            img_size=cfg.architecture.img_size,
            prototype_shape=tuple(cfg.architecture[arch].prototype_shape),
            radius=cfg.architecture.radius,
            num_classes=cfg.train.num_classes,
            prototype_activation_function=cfg.train.prototype_activation_function,
            sig_temp=cfg.train.sig_temp,
            add_on_layers_type=cfg.train.add_on_layers_type,
            layers= cfg.train.layers
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(str(device))
    ppnet = ppnet.to(device)
    model_ema = None
    class_specific = True

    # Optimizers
    joint_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': cfg.train.joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': cfg.train.joint_optimizer_lrs['prototype_vectors']},
    ]

    joint_optimizer = torch.optim.AdamW(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=cfg.train.joint_lr_step_size, gamma=0.1)

    warm_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': cfg.train.warm_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': cfg.train.warm_optimizer_lrs['prototype_vectors'],},
    ]
    warm_optimizer = torch.optim.AdamW(warm_optimizer_specs)

    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': cfg.train.last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.AdamW(last_layer_optimizer_specs)

    # Loss coefficients
    coefs = cfg.train.coefs.copy()  # Make a copy to avoid modifying config

    log('start training')
    log(f'Initial weight coefs: {coefs}')

    # =====================================================
    # STAGE 1 & 2: WARM-UP + JOINT TRAINING
    # =====================================================
    best_accu = 0.0
    best_epoch = -1
    best_model_path = os.path.join(model_dir, 'best_joint_model.pth')

    log('=' * 60)
    log('STAGE 1 & 2: WARM-UP AND JOINT TRAINING')
    log('=' * 60)

    for epoch in range(num_train_epochs):
        log('epoch: \t{0}'.format(epoch))

        if epoch < cfg.train.num_warm_epochs:
            aug_strength = 1.0
        elif epoch < cfg.train.num_warm_epochs + 10:
            aug_strength = 1.0
        else:
            aug_strength = 1.0
        train_dataset.set_aug_strength(aug_strength)

        if epoch < cfg.train.num_warm_epochs:
            ppnet.warmup = False
            ppnet.istraining = True
            tnt.warm_only(model=ppnet, log=log)
            train_acc, train_loss = tnt.train(
                model=ppnet, 
                dataloader=train_loader, 
                optimizer=warm_optimizer,
                class_specific=class_specific, 
                coefs=coefs, 
                log=log, 
                ema=model_ema, 
                clst_k=cfg.train.k, 
                sum_cls=cfg.train.sum_cls
            )

            # with torch.no_grad():
                # Take one batch from train_loader
                # for x_batch in train_loader:
                #     _, _, max_activation_slots = ppnet(x_batch['bmode'].to('cuda'))
                #     proto_scores = max_activation_slots.mean(0)  # [num_prototypes]
                #     for c in range(ppnet.num_classes):
                #         start = c * ppnet.num_prototypes_per_class
                #         end = (c + 1) * ppnet.num_prototypes_per_class
                #         print(f"Class {c} proto mean:", proto_scores[start:end].mean().item())
                #     break 

        else:
            # coefs = cfg.train.coefs
            ppnet.warmup = False
            ppnet.istraining = True
            tnt.joint(model=ppnet, log=log)
            train_acc, train_loss = tnt.train(
                model=ppnet, 
                dataloader=train_loader, 
                optimizer=joint_optimizer,
                class_specific=class_specific, 
                coefs=coefs, 
                log=log, 
                ema=model_ema, 
                clst_k=cfg.train.k, 
                sum_cls=cfg.train.sum_cls
            )
            joint_lr_scheduler.step()

            # with torch.no_grad():
            #     # Take one batch from train_loader
            #     for x_batch in train_loader:
            #         _, _, max_activation_slots = ppnet(x_batch['bmode'].to('cuda'))
            #         proto_scores = max_activation_slots.mean(0)  # [num_prototypes]
            #         for c in range(ppnet.num_classes):
            #             start = c * ppnet.num_prototypes_per_class
            #             end = (c + 1) * ppnet.num_prototypes_per_class
            #             print(f"Class {c} proto mean:", proto_scores[start:end].mean().item())
            #         break 
        ppnet.istraining = False
        accu, val_loss = tnt.test(
            model=ppnet, 
            dataloader=val_loader,
            class_specific=class_specific, 
            log=log, 
            clst_k=cfg.train.k,
            sum_cls=cfg.train.sum_cls
        )

        # Track best model
        if accu > best_accu:
            best_accu = accu
            best_epoch = epoch
            torch.save(ppnet.state_dict(), best_model_path)
            log(f'*** New best model saved with accuracy: {best_accu:.2f}% at epoch {best_epoch} ***')

        # Log to wandb
        phase = "warm" if epoch < cfg.train.num_warm_epochs else "joint"
        current_lr = warm_optimizer.param_groups[0]['lr'] if epoch < cfg.train.num_warm_epochs else joint_optimizer.param_groups[0]['lr']
        
        wandb.log({
            "epoch": epoch,
            "phase": phase,
            "train/accuracy": train_loss["acc"],
            # "train/cross_entropy": train_loss["cross entropy Loss"],
            "train/focal_loss": train_loss["focal loss"],
            "train/cluster_loss": train_loss["clst loss"],
            "train/separation_loss": train_loss["sep loss"],
            "train/orthogonal_loss": train_loss["orth loss"],
            "train/coherence_loss": train_loss.get("coherence loss", 0),
            "train/l1_loss": train_loss["l1 loss"],
            "train/total loss": train_loss["total loss"],
            # "val/cross entropy Loss": coefs['crs_ent']*val_loss["cross entropy Loss"],
            "val/focal loss": coefs['fcl']*val_loss["focal loss"],
            "val/cluster loss": coefs['clst']*val_loss["clst loss"],
            "val/separation loss": coefs['sep']*val_loss["sep loss"],
            "val/l1 loss": coefs['l1']*val_loss["l1 loss"],
            "val/orthogonal loss": coefs['orth']*val_loss["orth loss"],
            "val/coherence loss": coefs['coh']*val_loss.get("coherence loss", 0),
            "val/accuracy": val_loss["acc"],
            "accuracy": accu,
            "learning_rate": current_lr,
            "best_accuracy": best_accu,
        })

        save.save_model_w_condition(
            model=ppnet, 
            model_dir=model_dir, 
            model_name=str(epoch) + 'nopush', 
            accu=accu,
            target_accu=0.75, 
            log=log
        )

    # Load best model from joint training
    log('=' * 60)
    log(f'Loading best joint model from epoch {best_epoch} with accuracy {best_accu:.2f}%')
    log('=' * 60)
    ppnet.load_state_dict(torch.load(best_model_path))

    # =====================================================
    # STAGE 3: SLOTS PRUNING
    # =====================================================
    # log('=' * 60)
    # log('STAGE 3: SLOTS PRUNING')
    # log('=' * 60)

    # # Create optimizer for slots training (only now, not at the beginning)
    # slots_optimizer_specs = [{'params': ppnet.patch_select, 'lr': cfg.train.stage_2_lrs['patch_select']}]
    # slots_optimizer = torch.optim.AdamW(slots_optimizer_specs)
    # slots_lr_scheduler = torch.optim.lr_scheduler.StepLR(slots_optimizer, step_size=cfg.train.joint_lr_step_size, gamma=0.1)

    # # Update coherence loss coefficient for slots pruning
    # coefs['coh'] = cfg.train.coefs_slots['coh']
    # log(f'Coefs for slots training: {coefs}')

    # # Log initial slot statistics
    # with torch.no_grad():
    #     initial_slots = torch.sigmoid(ppnet.patch_select * ppnet.temp)
    #     initial_active = (initial_slots > 0.5).sum().item()
    #     total_slots = ppnet.patch_select.numel()
    #     log(f'Initial active slots: {initial_active}/{total_slots} ({100*initial_active/total_slots:.1f}%)')

    # for epoch in range(cfg.train.slots_train_epoch):
    #     tnt.joint(model=ppnet, log=log)  # All parameters trainable, but optimizer only updates patch_select
    #     log('slots epoch: \t{0}'.format(epoch))
        
    #     train_acc, train_loss = tnt.train(
    #         model=ppnet, 
    #         dataloader=train_loader, 
    #         optimizer=slots_optimizer,
    #         class_specific=class_specific, 
    #         coefs=coefs, 
    #         log=log, 
    #         ema=model_ema, 
    #         clst_k=cfg.train.k, 
    #         sum_cls=cfg.train.sum_cls
    #     )
    #     slots_lr_scheduler.step()
        
    #     accu, val_loss = tnt.test(
    #         model=ppnet, 
    #         dataloader=val_loader,
    #         class_specific=class_specific, 
    #         log=log, 
    #         clst_k=cfg.train.k, 
    #         sum_cls=cfg.train.sum_cls
    #     )

    #     # Log slot statistics
    #     with torch.no_grad():
    #         current_slots = torch.sigmoid(ppnet.patch_select * ppnet.temp)
    #         current_active = (current_slots > 0.5).sum().item()
            
    #     wandb.log({
    #         "epoch": num_train_epochs + epoch,
    #         "phase": "slots",
    #         "train/accuracy": train_loss["acc"],
    #         "train/cross_entropy": train_loss["cross entropy Loss"],
            # "train/focal_loss": train_loss["focal loss"],
    #         "train/cluster_loss": train_loss["clst loss"],
    #         "train/separation_loss": train_loss["sep loss"],
    #         "train/orthogonal_loss": train_loss["orth loss"],
    #         "train/coherence_loss": train_loss.get("coherence loss", 0),
    #         "train/l1_loss": train_loss["l1 loss"],
    #         "train/total loss": train_loss["total loss"],
    #         "val/cross entropy Loss": coefs['crs_ent']*val_loss["cross entropy Loss"],
            # "val/focal loss": coefs['fcl']*val_loss["focal loss"],
    #         "val/cluster loss": coefs['clst']*val_loss["clst loss"],
    #         "val/separation loss": coefs['sep']*val_loss["sep loss"],
    #         "val/l1 loss": coefs['l1']*val_loss["l1 loss"],
    #         "val/orthogonal loss": coefs['orth']*val_loss["orth loss"],
    #         "val/coherence loss": coefs['coh']*val_loss.get("coherence loss", 0),
    #         "val/accuracy": val_loss["acc"],
    #         "accuracy": accu,
    #         "slots/num_active": current_active,
    #         "slots/percentage_active": 100 * current_active / total_slots,
    #     })
        
    #     save.save_model_w_condition(
    #         model=ppnet, 
    #         model_dir=model_dir, 
    #         model_name=str(epoch) + 'slots', 
    #         accu=accu,
    #         target_accu=0.75, 
    #         log=log
    #     )

    # # Round slot indicators to binary values
    # log('=' * 60)
    # log('Rounding slot indicators to binary values')
    # log('=' * 60)
    
    # with torch.no_grad():
    #     slots_before_rounding = torch.sigmoid(ppnet.patch_select * ppnet.temp)
    #     active_before = (slots_before_rounding > 0.5).sum().item()
        
    #     # Round to binary
    #     ppnet.patch_select.data = torch.round(slots_before_rounding)
        
    #     active_after = (ppnet.patch_select > 0.5).sum().item()
    #     pruned = active_before - active_after
        
    # log(f'Slots before rounding: {active_before}/{total_slots} ({100*active_before/total_slots:.1f}%)')
    # log(f'Slots after rounding: {active_after}/{total_slots} ({100*active_after/total_slots:.1f}%)')
    # log(f'Slots pruned: {pruned}')
    
    # wandb.log({
    #     "slots/final_active": active_after,
    #     "slots/final_percentage": 100 * active_after / total_slots,
    #     "slots/num_pruned": pruned,
    # })

    # # Freeze slot indicators
    # ppnet.patch_select.requires_grad = False
    # log('Slot indicators frozen')

    # =====================================================
    # STAGE 4: PROTOTYPE PROJECTION
    # =====================================================
    log('=' * 60)
    log('STAGE 4: PROTOTYPE PROJECTION')
    log('=' * 60)

    push_greedy.push_prototypes(
        train_push_loader,
        pnet=ppnet,
        class_specific=class_specific,
        preprocess_input_function=preprocess_input_function,
        prototype_layer_stride=1,
        root_dir_for_saving_prototypes=img_dir,
        epoch_number='final',
        prototype_img_filename_prefix=prototype_img_filename_prefix,
        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
        save_prototype_class_identity=True,
        log=log
    )
    
    accu, val_loss = tnt.test(
        model=ppnet, 
        dataloader=val_loader,
        class_specific=class_specific, 
        log=log, 
        clst_k=cfg.train.k, 
        sum_cls=cfg.train.sum_cls
    )
    
    log(f'Accuracy after projection: {accu:.2f}%')
    
    save.save_model_w_condition(
        model=ppnet, 
        model_dir=model_dir, 
        model_name='push', 
        accu=accu,
        target_accu=0.0, 
        log=log
    )

    # =====================================================
    # STAGE 5: LAST LAYER OPTIMIZATION
    # =====================================================
    # log('=' * 60)
    # log('STAGE 5: LAST LAYER OPTIMIZATION')
    # log('=' * 60)

    # num_finetune_epochs = cfg.train.get('num_finetune_epochs', 15)  # Get from config, default to 15
    
    # for epoch in range(num_finetune_epochs):
    #     tnt.last_only(model=ppnet, log=log)
    #     log('finetune iteration: \t{0}'.format(epoch))
        
    #     train_acc, train_loss = tnt.train(
    #         model=ppnet, 
    #         dataloader=train_loader, 
    #         optimizer=last_layer_optimizer,
    #         class_specific=class_specific, 
    #         coefs=coefs, 
    #         log=log, 
    #         ema=model_ema, 
    #         clst_k=cfg.train.k, 
    #         sum_cls=cfg.train.sum_cls
    #     )
        
    #     accu, val_loss = tnt.test(
    #         model=ppnet, 
    #         dataloader=val_loader,
    #         class_specific=class_specific, 
    #         log=log, 
    #         clst_k=cfg.train.k,
    #         sum_cls=cfg.train.sum_cls
    #     )
        
    #     wandb.log({
    #         "epoch": num_train_epochs + cfg.train.slots_train_epoch + epoch,
    #         "phase": "finetune",
    #         "train/accuracy": train_loss["acc"],
    #         "train/cross_entropy": train_loss["cross entropy Loss"],
            # "train/focal_loss": train_loss["focal loss"],
    #         "train/cluster_loss": train_loss["clst loss"],
    #         "train/separation_loss": train_loss["sep loss"],
    #         "train/orthogonal_loss": train_loss["orth loss"],
    #         "train/coherence_loss": train_loss.get("coherence loss", 0),
    #         "train/l1_loss": train_loss["l1 loss"],
    #         "train/total loss": train_loss["total loss"],
    #         "val/cross entropy Loss": coefs['crs_ent']*val_loss["cross entropy Loss"],
            # "val/focal loss": coefs['fcl']*val_loss["focal loss"],
    #         "val/cluster loss": coefs['clst']*val_loss["clst loss"],
    #         "val/separation loss": coefs['sep']*val_loss["sep loss"],
    #         "val/l1 loss": coefs['l1']*val_loss["l1 loss"],
    #         "val/orthogonal loss": coefs['orth']*val_loss["orth loss"],
    #         "val/coherence loss": coefs['coh']*val_loss.get("coherence loss", 0),
    #         "val/accuracy": val_loss["acc"],
    #         "accuracy": accu,
    #     })
        
    #     save.save_model_w_condition(
    #         model=ppnet, 
    #         model_dir=model_dir, 
    #         model_name=str(epoch) + 'finetuned', 
    #         accu=accu,
    #         target_accu=0.70, 
    #         log=log
    #     )

    log('=' * 60)
    log('TRAINING COMPLETE')
    log('=' * 60)
    log(f'Best joint training accuracy: {best_accu:.2f}% at epoch {best_epoch}')
    log(f'Final accuracy after all stages: {accu:.2f}%')
    
    logclose()


def load_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '-c', help='Path to config file')
    parser.add_argument('-gpuid', nargs=1, type=str, default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    cfg = load_config(args.config)
    main(cfg)