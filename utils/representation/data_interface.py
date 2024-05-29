import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .load_dataset import DatasetTask


class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, num_workers=8, data_task=None, split='train', *args, **kwargs):
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, *args, **kwargs)
        self.pretrain_device = 'cuda:0'
        self.split = split
        self.data_task = data_task

    def __iter__(self):
        for batch in super().__iter__():
            try:
                self.pretrain_device = f'cuda:{torch.distributed.get_rank()}'
            except:
                self.pretrain_device = 'cuda:0'

            stream = torch.cuda.Stream(
                self.pretrain_device
            )
            with torch.cuda.stream(stream):
                sample = {}
                for key in batch[0].keys():
                    sample[key] = [one[key] for one in batch]
                    if type(sample[key][0]) == torch.Tensor:
                        sample[key] = torch.stack(sample[key], dim=0).cuda(non_blocking=True, device=self.pretrain_device)

                yield sample


def collate_fn(batch):
    return batch


class DInterface_base(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = self.hparams.batch_size
        print("batch_size", self.batch_size)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.data_task.datasets['train']
            self.valset = self.data_task.datasets['valid']

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.data_task.datasets['test']

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, prefetch_factor=3)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class DInterface(DInterface_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.data_task = DatasetTask(self.hparams)
        self.data_task.load_dataset_mix_enc_dec('train')
        self.data_task.load_dataset_mix_enc_dec('valid')
        self.data_task.load_dataset_mix_enc_dec('test')

    def train_dataloader(self):
        return MyDataLoader(self.trainset, batch_size=self.batch_size, split='train', num_workers=self.hparams.num_workers, data_task=self.data_task, shuffle=True, prefetch_factor=8, pin_memory=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return MyDataLoader(self.testset, batch_size=self.batch_size, split='valid', num_workers=self.hparams.num_workers, data_task=self.data_task, shuffle=False, pin_memory=True, collate_fn=collate_fn)

    def test_dataloader(self):
        return MyDataLoader(self.testset, batch_size=self.batch_size, split='test', num_workers=self.hparams.num_workers, data_task=self.data_task, shuffle=False, pin_memory=True, collate_fn=collate_fn)
