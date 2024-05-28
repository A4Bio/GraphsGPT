import inspect
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs


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
