import os
import argparse

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .datamodules import return_prepared_dm
from lit_vit.models.lit_visiontransformer import LitVisionTransformer

def ret_args(ret_parser=False):

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help='random seed for initialization')
    parser.add_argument('--no_cpu_workers', type=int, default=4, help='CPU workers for data loading.')

    parser.add_argument('--results_dir', default='results_training', type=str,
                        help='The directory where results will be stored')
    parser.add_argument('--save_checkpoint_freq', default=100, type=int,
                        help='Frequency (in epochs) to save checkpoints')

    parser.add_argument('--dataset_name', choices=['cifar10', 'cifar100', 'imagenet'], 
                        default='cifar10', help='Which dataset to use.')
    parser.add_argument('--dataset_path', help='Path for the dataset.')
    parser.add_argument("--deit_recipe", action='store_true',
                        help="Use DeiT training recipe")
    
    parser.add_argument('--image_size', default=224, type=int,
                        help='Image (square) resolution size')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for train/val/test.')
    
    parser = LitVisionTransformer.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=1, max_epochs=2, gradient_clip_val=1.0)
    parser.set_defaults(precision=32, log_gpu_memory=None, profiler=None)

    if ret_parser:
        return parser
    args = parser.parse_args()

    args.run_name = '{}_{}_is{}_bs{}_{}lr{}wd{}_pt{}_seed{}'.format(
    args.dataset_name, args.model_name, args.image_size, args.batch_size, 
    args.optimizer, args.learning_rate, args.weight_decay, args.pretrained, args.seed)

    if args.deit_recipe:
        ''' taken from DeiT paper
        https://arxiv.org/abs/2012.12877
        https://github.com/facebookresearch/deit/blob/main/main.py'''
        # augmentation and random erase params
        args.color_jitter = 0.4
        args.aa = 'rand-m9-mstd0.5-inc1'
        args.smoothing = 0.1
        args.train_interpolation = 'bicubic'
        args.repeated_aug = True
        args.reprob = 0.25
        args.remode = 'pixel'
        args.recount = 1
        args.resplit = False
        # mixup params
        args.mixup = 0.8
        args.cutmix = 1.0
        args.cutmix_minmax = None
        args.mixup_prob = 1.0
        args.mixup_switch_prob = 0.5
        args.mixup_mode = 'batch'

    return args


def load_trainer(args, dm, wandb_logger):
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.results_dir, args.run_name), 
        filename='{epoch}', monitor='val_loss', verbose=True, save_last=True,
        save_top_k=1, save_weights_only=True, mode='min')
    
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
    trainer.logger = wandb_logger
    
    if trainer.max_steps:
        total_steps = trainer.max_steps
    else:
        total_steps = trainer.max_epochs * len(dm.train_dataloader())
    args.total_steps = total_steps

    return trainer


def environment_loader(args, init=True):
        
    # set up W&B logger
    if init:
        os.makedirs(args.results_dir, exist_ok=True)
        wandb.init(config=args)
        wandb.run.name = args.run_name
        wandb_logger = WandbLogger(name=args.run_name)
        
    # seed everything
    pl.seed_everything(seed=args.seed)

    # prepare datamodule
    dm = return_prepared_dm(args)

    # setup model and trainer
    trainer = load_trainer(args, dm, wandb_logger)
    model = LitVisionTransformer(args)

    return dm, trainer, model
