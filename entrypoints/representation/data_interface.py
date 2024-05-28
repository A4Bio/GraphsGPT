import torch
from torch.utils.data import DataLoader

from entrypoints.representation.utils.data_interface import DInterface_base
from entrypoints.representation.utils.load_dataset import DatasetTask


def prepare_dict(my_dict):
    new_dict = {}
    for key, value in my_dict.items():
        # 切割键
        key_parts = key.split('.')
        if len(key_parts) == 1:
            first_level_key = key_parts[0]
            new_dict[first_level_key] = value
        elif len(key_parts) == 2:
            # 获取第一级键和第二级键
            first_level_key = key_parts[0]
            second_level_key = key_parts[1]

            # 检查第一级键是否已经存在于新字典中
            if first_level_key in new_dict:
                # 如果已存在，将值添加到第二级键下
                new_dict[first_level_key][second_level_key] = value
            else:
                # 如果不存在，创建第一级键并添加第二级键和对应的值
                new_dict[first_level_key] = {second_level_key: value}
    return new_dict


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

                # for k, ds in self.dataset.defn.items():
                #     if self.split == 'train':
                #         sample[k] = ds.collater([pair[0][k] for pair in batch]+[pair[1][k] for pair in batch])
                #     else:
                #         sample[k] = ds.collater([pair[0][k] for pair in batch])

                #     if type(sample[k]) == torch.Tensor:
                #         sample[k] = sample[k].cuda(non_blocking=True, device=self.pretrain_device)

                # sample = prepare_dict(sample)
                yield sample


def collate_fn(batch):
    return batch


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
