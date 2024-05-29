import os

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from .dictionary import Dictionary
from .lmdb_dataset import LMDBDataset

BASE_PATH = os.path.dirname(__file__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
}


def batch_generator(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


from torch.utils.data import Dataset


class MixDataset(Dataset):
    def __init__(self, data, strategy, split, self_prob):
        self.data = data  # Assuming data is a list or numpy array containing your dataset
        self.strategy = strategy
        self.split = split
        self.self_prob = self_prob

    def __len__(self):
        key = list(self.data.keys())[0]
        return len(self.data[key])

    def __getitem__(self, index):
        if self.split == 'train':
            if self.strategy == 'mix_graph':
                if np.random.random() <= self.self_prob:
                    sample = {'mol_raw': self.data['mol_raw'][index],
                              'target_raw': self.data['target_raw'][index]}
                else:
                    idx = np.random.choice(len(self.data['mol_mix']), 1).item()
                    sample = {'mol_raw': self.data['mol_mix'][idx],
                              'target_raw': self.data['target_mix'][idx]}
            else:
                sample = {key: val[index] for key, val in self.data.items()}
        else:
            sample = {key: val[index] for key, val in self.data.items()}

        sample['smi_raw'] = Chem.MolToSmiles(sample['mol_raw'], canonical=True, isomericSmiles=False)
        return sample


class DatasetTask:
    """Task for training transformer auto-encoder models."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dictionary = Dictionary.load(os.path.join(BASE_PATH, "dict.txt"))
        self.seed = args.seed
        self.args.count = bool(self.args.count)
        self.args.multiple_conformation = bool(self.args.multiple_conformation)
        # add mask token
        self.mask_idx = self.dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]

        from .fingerprint_interface import FPModel
        self.FPModel = FPModel(args.fingerprint)
        self.datasets = {}
        self.defns = {}
        self.beta_distribution = torch.distributions.beta.Beta(args.mixup_alpha, args.mixup_alpha)

    def check_dataset(self, dataset):
        is_valid, smiles_list = {}, [data['smi'] for data in dataset]
        for smiles in smiles_list:
            token = self.FPModel.tokenizer.encode_plus(smiles, return_tensors="pt")
            if token is None:
                is_valid[smiles] = False
            elif token['graph_position_ids_1'].shape[0] > 256:
                is_valid[smiles] = False
            else:
                is_valid[smiles] = True
        return is_valid

    def get_fps(self, fingerprint, dataset):
        if fingerprint == 'graphsgpt':
            smiles_list = [data['smi'] for data in dataset]
            is_valid = self.check_dataset(dataset)

            fingerprint_embeddings = []
            for batch in batch_generator(smiles_list, 128):
                valid_smiles = [smiles for smiles in batch if is_valid[smiles]]
                valid_embeds = self.FPModel(valid_smiles)
                final_embeds = torch.zeros(len(batch), valid_embeds.shape[1])
                valid_idx = torch.tensor([is_valid[smiles] for smiles in batch]).nonzero().view(-1)
                final_embeds[valid_idx] = valid_embeds
                fingerprint_embeddings.append(final_embeds)

            fingerprint_embeddings = torch.cat(fingerprint_embeddings, dim=0)

        if fingerprint == 'morgan':
            smiles_list = [data['smi'] for data in dataset]
            fingerprint_embeddings = self.FPModel(smiles_list)
        return fingerprint_embeddings

    def get_mixup_graph(self, fingerprint, K, self_prob, targets, device='cuda'):
        mixup_lam = torch.tensor([self.beta_distribution.rsample((1,)).item() for i in range(fingerprint.shape[0])]).view(-1, 1)

        dist = torch.cdist(fingerprint, fingerprint, p=2)
        dist = (dist - dist.mean(dim=0)) / dist.std(dim=0)  # morgan: [-3.7, 1.9], graphsgpt: [-3.7, 2.0]

        MIN = -9999999999
        invalid_mask = (fingerprint == 0).all(dim=1)
        dist = torch.exp(-dist)
        dist[torch.eye(dist.shape[0], dtype=torch.bool)] = MIN
        dist[invalid_mask] = MIN
        dist[:, invalid_mask] = MIN
        # dist[torch.eye(dist.shape[0], dtype=torch.bool)] = MIN
        dist = dist.float()
        prob = torch.softmax(dist, dim=-1)
        # prob[torch.eye(dist.shape[0], dtype=torch.bool)] = self_prob
        # prob[~torch.eye(dist.shape[0], dtype=torch.bool)] = (1 - self_prob)*prob[~torch.eye(dist.shape[0], dtype=torch.bool)]

        mol_mixed_all, targets_mixed_all, mixup_lam_all = [], [], []
        # pair_index = torch.multinomial(prob, 1).view(-1)
        val, pair_index = torch.topk(prob, K + 1, dim=-1)
        pair_index = pair_index[:, -1]

        if self.args.loss_type == 'mixup_multi_task_BCE':
            valid = (targets > -0.5) & (targets[pair_index] > -0.5)
            targets_mixed = mixup_lam * targets + (1 - mixup_lam) * targets[pair_index]
            targets_mixed = targets_mixed * valid + (-1) * (~valid)
        else:
            targets_mixed = mixup_lam * targets + (1 - mixup_lam) * targets[pair_index]
        fingerprints_mixed = mixup_lam * fingerprint + (1 - mixup_lam) * fingerprint[pair_index]
        fingerprints_mixed = fingerprints_mixed[:, None, :].to(device)

        for idx in range(0, fingerprints_mixed.shape[0], 1024):
            all_results = self.FPModel.model.generate_from_fingerprints(
                fingerprints_mixed[idx:idx + 1024],
                bond_dict=self.FPModel.tokenizer.bond_dict,
                max_atoms=None,
                similarity_threshold=0.5,
                check_begin_similarity=True,
                check_first_node=True,
                use_cache=False,
                save_failed=True,
                device=device,
                verbose=False,
            )

            for k, result in enumerate(all_results):
                if result is not None:
                    mol, smiles = self.FPModel.tokenizer.decode(result)
                    if mol is None:
                        # mol_mixed.append(None)
                        continue
                    mol_mixed_all.append(mol)
                    targets_mixed_all.append(targets_mixed[idx + k])
                    mixup_lam_all.append(mixup_lam[idx + k])

        return mol_mixed_all, targets_mixed_all, mixup_lam_all

    def get_mixup_embed(self, fingerprint, temperature, self_prob, targets, device='cuda'):
        mixup_lam = torch.tensor([self.beta_distribution.rsample((1,)).item() for i in range(fingerprint.shape[0])]).view(-1, 1)

        dist = torch.cdist(fingerprint, fingerprint, p=2)
        dist = (dist - dist.mean(dim=0)) / dist.std(dim=0)  # morgan: [-3.7, 1.9], graphsgpt: [-3.7, 2.0]

        MIN = -9999999999
        invalid_mask = (fingerprint == 0).all(dim=1)
        dist = torch.exp(-dist)
        dist[torch.eye(dist.shape[0], dtype=torch.bool)] = MIN
        dist[invalid_mask] = MIN
        dist[:, invalid_mask] = MIN
        dist = dist.float()
        prob = torch.softmax(dist / temperature, dim=-1)
        prob[torch.eye(dist.shape[0], dtype=torch.bool)] = self_prob
        prob[~torch.eye(dist.shape[0], dtype=torch.bool)] = (1 - self_prob) * prob[~torch.eye(dist.shape[0], dtype=torch.bool)]

        pair_index = torch.multinomial(prob, 1).view(-1)
        return pair_index, mixup_lam

    def load_dataset_mix_enc_dec(self, split):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        # self.fingerprint_embeddings = self.gen_fingerprint(dataset)

        if os.path.exists(os.path.join(self.args.data, self.args.task_name, f'fingerprint_{split}.pt')):
            fingerprint_embeddings = torch.load(os.path.join(self.args.data, self.args.task_name, f'fingerprint_{split}.pt'))
        else:
            fingerprint_embeddings = self.get_fps(self.args.fingerprint, dataset)
            torch.save(fingerprint_embeddings, os.path.join(self.args.data, self.args.task_name, f'fingerprint_{split}.pt'))
        self.__dict__[f'fingerprint_embeddings_{split}'] = fingerprint_embeddings

        mols_raw = [Chem.MolFromSmiles(one['smi']) for one in dataset]
        targets_raw = torch.tensor([one['target'] for one in dataset])
        fingerprints = self.__dict__[f'fingerprint_embeddings_{split}']

        # ========== ignore invalid molecules ==============
        invalid = (fingerprints == 0).all(dim=1)
        print(f'ignore {invalid.sum()} molecules in {self.args.task_name}! The split is {split}.')
        mols_raw = [one for i, one in enumerate(mols_raw) if not invalid[i]]
        targets_raw = targets_raw[~invalid]
        fingerprints = fingerprints[~invalid]

        if self.args.task_name in ['bace', 'bbbp', 'clintox', 'hiv']:
            targets_raw = F.one_hot(targets_raw.view(-1), num_classes=2).float()

        if self.args.mixup_strategy in ['no_mix_pretrain', 'no_mix_vanilla']:
            mols_mixed = mols_raw
            targets_mixed = targets_raw
            mixup_lam = torch.ones(len(mols_raw)).view(-1, 1)

        if self.args.mixup_strategy in ['mix_graph']:
            if split == 'train':
                mols_mixed, targets_mixed, mixup_lam = [], [], []
                for k in range(self.args.mix_times):
                    mols_mixed_batch, targets_mixed_batch, mixup_lam_batch = self.get_mixup_graph(fingerprints, 1, self.args.self_prob, targets_raw)
                    mols_mixed.append(mols_mixed_batch)
                    targets_mixed.append(targets_mixed_batch)
                    mixup_lam.append(mixup_lam_batch)
                mols_mixed = list(sum(mols_mixed, []))
                targets_mixed = torch.stack(list(sum(targets_mixed, [])))
                mixup_lam = torch.stack(list(sum(mixup_lam, [])))
                # for i in range(len(mols_raw)):
                #     if mols_mixed[i] is None:
                #         mols_mixed[i] = mols_raw[i]
                #         targets_mixed[i] = targets_raw[i]
            else:
                mols_mixed = mols_raw
                targets_mixed = targets_raw
                mixup_lam = torch.ones(len(mols_raw)).view(-1, 1)

        if self.args.mixup_strategy in ['mix_embed']:
            if split == 'train':
                pair_index, mixup_lam = self.get_mixup_embed(self.__dict__[f'fingerprint_embeddings_{split}'], self.args.mixup_temperature, self.args.self_prob, targets_raw)
                mols_mixed, targets_mixed = [], []
                for i in range(len(mols_raw)):
                    mols_mixed.append(mols_raw[pair_index[i]])
                    targets_mixed.append(targets_raw[pair_index[i]])
            else:
                mols_mixed = mols_raw
                targets_mixed = targets_raw
                mixup_lam = torch.ones(len(mols_raw)).view(-1, 1)

        mixup_dataset = {'mol_raw': mols_raw,
                         'target_raw': targets_raw,
                         'mol_mix': mols_mixed,
                         'target_mix': targets_mixed,
                         'mixup_lam': mixup_lam}

        self.datasets[split] = MixDataset(mixup_dataset, self.args.mixup_strategy, split, self.args.self_prob)
