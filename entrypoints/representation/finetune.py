import argparse
import datetime
import math
import os
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
import sys
import torch
import warnings
from omegaconf import OmegaConf
from pytorch_lightning.trainer import Trainer

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

from utils.representation.logger import SetupCallback
from utils.representation.data_interface import DInterface
from utils.representation.model_interface import MInterface

model_zoom = {
    'tox21': ['./data_finetune/model_zoom/tox21/tox21.yaml',
              './data_finetune/model_zoom/tox21/tox21.pth'],
    'toxcast': ['./data_finetune/model_zoom/toxcast/toxcast.yaml',
                './data_finetune/model_zoom/toxcast/toxcast.pth'],
    'bbbp': ['./data_finetune/model_zoom/bbbp/bbbp.yaml',
             './data_finetune/model_zoom/bbbp/bbbp.pth'],
    'sider': ['./data_finetune/model_zoom/sider/sider.yaml',
              './data_finetune/model_zoom/sider/sider.pth'],
    'hiv': ['./data_finetune/model_zoom/hiv/hiv.yaml',
            './data_finetune/model_zoom/hiv/hiv.pth'],
    'bace': ['./data_finetune/model_zoom/bace/bace.yaml',
             './data_finetune/model_zoom/bace/bace.pth']
}


def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default="./results/representation", type=str)
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)

    parser.add_argument('--task_name', default='bace', choices=['bace', 'bbbp', 'clintox', 'sider', 'tox21', 'toxcast', 'hiv', 'esol', 'freesolv', 'lipo'])
    parser.add_argument('--mixup_strategy', default='no_mix_pretrain', choices=['mix_embed', 'mix_graph', 'no_mix_pretrain', 'no_mix_vanilla'])
    parser.add_argument('--lr_scheduler', default='cosine')
    parser.add_argument('--offline', default=0, type=int)

    # dataset parameters
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--count', default=0, type=int)
    parser.add_argument('--multiple_conformation', default=1, type=int)
    parser.add_argument('--only_polar', default=-1, type=int)
    parser.add_argument('--remove_polar_hydrogen', default=False, type=bool)
    parser.add_argument('--remove_hydrogen', default=False, type=bool)
    parser.add_argument('--fingerprint', default='graphsgpt', type=str)
    parser.add_argument('--data', default='./data_finetune/molecular_property_prediction', type=str)
    parser.add_argument('--conf_size', default=11, type=int)
    parser.add_argument('--max_atoms', default=256, type=int)
    parser.add_argument('--self_prob', default=0.1, type=float)
    parser.add_argument('--no_shuffle', default=False, type=bool)
    parser.add_argument('--mix_times', default=1, type=int)

    # Training parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch', default=50, type=int, help='end epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--warmup_ratio', default=0.06, type=float, help='warmup rate')
    parser.add_argument('--ema_decay', default=0.999, type=float, help='warmup rate')
    parser.add_argument('--pos_weight', default=99, type=float, help='warmup rate')

    # Model parameters
    parser.add_argument('--encoder_layers', default=15, type=int)
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--ffn_embed_dim', default=2048, type=int)
    parser.add_argument('--attention_heads', default=64, type=int)
    parser.add_argument('--emb_dropout', default=0.1, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--attention_dropout', default=0.1, type=float)
    parser.add_argument('--activation_dropout', default=0.0, type=float)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--activation_fn', default='gelu', type=str)
    parser.add_argument('--post_ln', default=False, type=bool)
    parser.add_argument('--no_final_head_layer_norm', default=True, type=bool)
    parser.add_argument('--mixup_alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--encoder_embed_dim', default=512, type=int)
    parser.add_argument('--pooler_dropout', default=0.0, type=float)
    parser.add_argument('--num_classes', default=2, type=int)  # need to be changed
    parser.add_argument('--loss_type', default='mixup_multi_task_BCE', type=str)  # need to be changed
    parser.add_argument('--checkpoint_metric', default='test_auc', type=str)  # need to be changed
    args = parser.parse_args()
    return args


def load_callbacks(args):
    callbacks = []

    logdir = str(os.path.join(args.res_dir, args.ex_name))

    ckptdir = os.path.join(logdir, "checkpoints")

    modes = {'test_auc': 'max', 'test_rmse': 'min', 'test_mae': 'min'}

    metric = args.checkpoint_metric
    sv_filename = 'best-{epoch:02d}-{' + metric + ':.3f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=5,
        mode=modes[metric],
        save_last=True,
        dirpath=ckptdir,
        verbose=True,
        every_n_epochs=args.check_val_every_n_epoch,
    ))

    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
            now=now,
            logdir=logdir,
            ckptdir=ckptdir,
            cfgdir=cfgdir,
            config=args.__dict__,
            argv_content=sys.argv + ["gpus: {}".format(torch.cuda.device_count())], )
    )

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks


def main():
    args = create_parser()
    params = OmegaConf.load(model_zoom[args.task_name][0])
    config = args.__dict__
    config.update(params)
    logger = plog.WandbLogger(
        project='graphsgpt_mixup2',
        name=args.ex_name,
        save_dir=str(os.path.join(args.res_dir, args.ex_name)),
        offline=True,
        id="_".join(args.ex_name.split("/")),
        entity="gaozhangyang"
    )

    pl.seed_everything(args.seed)

    data_module = DInterface(**vars(args))
    data_module.setup()

    gpu_count = torch.cuda.device_count()
    args.steps_per_epoch = math.ceil(len(data_module.trainset) / args.batch_size / gpu_count)
    print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")

    model = MInterface(**vars(args))
    params = torch.load(model_zoom[args.task_name][1])
    params = {k.replace('_forward_module.', ''): v for k, v in params.items()}
    model.load_state_dict(params)

    trainer_config = {
        'gpus': -1,  # Use all available GPUs
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'deepspeed_stage_2',  # 'ddp', 'deepspeed_stage_2
        "precision": '32',  # "bf16", 16
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': load_callbacks(args),
        'logger': logger,
        'gradient_clip_val': 1.0
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_opt)

    trainer.test(model, data_module)
    print(trainer_config)


if __name__ == "__main__":
    main()
