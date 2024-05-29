import inspect
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn as nn
from torch.optim import lr_scheduler as lrs

from .graphsgpt_finetune_model import GraphsGPT
from .load_dataset import task_metainfo


def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
    batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')


class MInterface_base(pl.LightningModule):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)

    def forward(self, input):
        pass

    def training_step(self, batch, batch_idx, **kwargs):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')

    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                   step_size=self.hparams.lr_decay_steps,
                                   gamma=self.hparams.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                              T_max=max(self.hparams.epoch / 5, 1))
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=self.hparams.steps_per_epoch, epochs=self.hparams.epoch, three_phase=False)
        elif lr_scheduler == 'polynomial':
            scheduler = PolynomialDecayLRSchedule(optimizer, warmup_ratio=self.hparams.warmup_ratio, total_num_update=self.hparams.epoch * self.hparams.steps_per_epoch, lr=self.hparams.lr, end_learning_rate=0.0, power=1.0)
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0

        optimizer_g = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay, betas=(0.9, 0.99), eps=1e-8)

        schecular_g = self.get_schedular(optimizer_g, self.hparams.lr_scheduler)

        return [optimizer_g], [{"scheduler": schecular_g, "interval": "step"}]

    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_devices(self):
        self.device = torch.device(self.hparams.device)

    def configure_loss(self):
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def load_model(self):
        self.model = None

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)


class MInterface(MInterface_base):
    def __init__(self, model_name=None, loss=None, lr=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.configure_loss()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        self.targets = []
        self.preds = []

    def forward(self, batch, mode='eval'):
        results = self.model(batch['mol_raw'])
        targets = batch["target_raw"]
        logit_output = results['logit_output']

        results_mix = self.compute_loss(logit_output, targets, loss_type=self.hparams.loss_type)
        loss = results_mix['loss']

        return {'loss': loss, 'logits': logit_output, 'targets': targets}

    def training_step(self, batch, batch_idx, **kwargs):
        results = self(batch, 'train')
        self.log_dict({"train_loss": results['loss']}, on_epoch=True, prog_bar=True)
        return results['loss']

    def validation_step(self, batch, batch_idx):
        results = self(batch, 'eval')
        self.targets.append(results['targets'].float().cpu().numpy())
        self.preds.append(results['logits'].float().cpu().numpy())
        metrics = self.compute_metrics(results)
        metrics.update({'loss': results['loss']})
        val_metrics = {'test_' + key: val for key, val in metrics.items()}
        self.log_dict(val_metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        return metrics

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        metrics = self.compute_metrics(outputs)
        val_metrics = {'test_' + key: val for key, val in metrics.items()}
        self.log_dict(val_metrics)

        self.targets = []
        self.preds = []
        return self.log_dict

    def test_epoch_end(self, outputs):
        metrics = self.compute_metrics(outputs)
        val_metrics = {'test_' + key: val for key, val in metrics.items()}
        self.log_dict(val_metrics)

        self.targets = []
        self.preds = []
        return self.log_dict

    def load_model(self):
        # =========== graphsgpt model
        self.model = GraphsGPT(self.hparams.task_name, self.hparams.mixup_strategy, self.hparams.pooler_dropout)

    def compute_metrics(self, outputs):
        targets = np.concatenate(self.targets)
        preds = np.concatenate(self.preds)

        if self.hparams.loss_type == 'mixup_cross_entropy':
            targets = targets.argmax(axis=-1)
            if self.hparams.num_classes == 2:
                if np.unique(targets).shape[0] == 1:
                    auc = 0.0
                else:
                    auc = roc_auc_score(targets, preds[:, 1])
                return {'auc': auc}

        if self.hparams.loss_type == 'mixup_mse':
            mean = task_metainfo[self.hparams.task_name]["mean"]
            std = task_metainfo[self.hparams.task_name]["std"]
            predicts = preds * std + mean
            mae = np.abs(predicts - targets).mean()
            mse = ((predicts - targets) ** 2).mean()
            return {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}

        if self.hparams.loss_type == 'mixup_smooth_mae':
            mean = task_metainfo[self.hparams.task_name]["mean"]
            std = task_metainfo[self.hparams.task_name]["std"]
            predicts = preds * std + mean
            mae = np.abs(predicts - targets).mean()
            mse = ((predicts - targets) ** 2).mean()
            return {'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse)}

        if self.hparams.loss_type == 'mixup_multi_task_BCE':
            def sigmoid(z):
                return 1 / (1 + np.exp(-z))

            probs = sigmoid(preds)
            agg_auc_list = []
            for i in range(targets.shape[1]):
                if np.sum(targets[:, i] == 1) > 0 and np.sum(targets[:, i] == 0) > 0:
                    # ignore nan values
                    is_labeled = targets[:, i] > -0.5
                    agg_auc_list.append(
                        roc_auc_score(targets[is_labeled, i], probs[is_labeled, i])
                    )

            auc = sum(agg_auc_list) / (len(agg_auc_list) + 1e-8)
            return {'auc': auc}

    def cross_entropy_loss(self, net_output, targets, reduce=True):
        targets = targets.view(-1)
        lprobs = F.log_softmax(net_output, dim=-1)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(
            lprobs,
            targets,
            reduction="sum" if reduce else "none",
        )

        return {'loss': loss}

    def multi_task_BCE(self, logit_output, targets, reduce=True):
        is_labeled = targets > -0.5
        pos_mask = targets[is_labeled].float().view(-1)
        loss = F.binary_cross_entropy_with_logits(
            logit_output[is_labeled].float().view(-1),
            targets[is_labeled].float().view(-1),
            reduction="sum" if reduce else "none",
            pos_weight=((1 - pos_mask) * 9 + 1)
        )
        return {'loss': loss}

    def mse(self, logit_output, targets, reduce=True):
        predicts_normed = logit_output.view(-1, self.hparams.num_classes).float()
        targets = (
            targets.view(-1, self.hparams.num_classes).float()
        )

        mean = task_metainfo[self.hparams.task_name]["mean"]
        std = task_metainfo[self.hparams.task_name]["std"]

        targets_mean = torch.tensor(mean, device=targets.device)
        targets_std = torch.tensor(std, device=targets.device)
        targets_normed = (targets - targets_mean) / targets_std
        loss = F.mse_loss(
            predicts_normed,
            targets_normed,
            reduction="sum" if reduce else "none",
        )
        return {'loss': loss}

    def smooth_mse(self, logit_output, targets, reduce=True):
        predicts_normed = logit_output.view(-1, self.hparams.num_classes).float()
        targets = (
            targets.view(-1, self.hparams.num_classes).float()
        )
        mean = task_metainfo[self.hparams.task_name]["mean"]
        std = task_metainfo[self.hparams.task_name]["std"]
        targets_mean = torch.tensor(mean, device=targets.device)
        targets_std = torch.tensor(std, device=targets.device)
        targets_normed = (targets - targets_mean) / targets_std
        loss = F.smooth_l1_loss(
            predicts_normed,
            targets_normed,
            reduction="sum" if reduce else "none",
        )
        return {'loss': loss}

    def compute_loss(self, logit_output, targets, loss_type='mixup_cross_entropy'):
        if loss_type == 'mixup_cross_entropy':
            loss = softmax_cross_entropy_with_softtarget(logit_output, targets, reduction='mean')
            return {'loss': loss}

        if loss_type == 'mixup_mse':
            return self.mse(logit_output, targets, reduce=True)

        if loss_type == 'mixup_smooth_mae':
            return self.smooth_mse(logit_output, targets, reduce=True)

        if loss_type == 'mixup_multi_task_BCE':
            return self.multi_task_BCE(logit_output, targets, reduce=True)


class PolynomialDecayLRSchedule(lrs.LRScheduler):
    def __init__(self, optimizer, warmup_ratio, total_num_update, lr, end_learning_rate, power, last_epoch=-1):
        self.warmup_ratio = warmup_ratio  # 2532
        self.warmup_updates = int(self.warmup_ratio * total_num_update)
        self.total_num_update = total_num_update  # 42200
        self.lr = lr
        self.warmup_factor = 1.0 / self.warmup_updates if self.warmup_updates > 0 else 1
        self.end_learning_rate = end_learning_rate
        self.power = power
        super(PolynomialDecayLRSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            return [self.warmup_factor * self.lr]
        elif self.last_epoch >= self.total_num_update:
            return [self.end_learning_rate]
        else:
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (self._step_count - self.warmup_updates) / (self.total_num_update - self.warmup_updates)
            lr = lr_range * pct_remaining ** self.power + self.end_learning_rate
            return [lr]
