from multiprocessing import Pool

import os
import pickle
import torch
import torch.nn.utils.rnn as rnn_utils
from rdkit import Chem, RDLogger
from rdkit.Chem import MolStandardize, AllChem
from transformers import PreTrainedTokenizer
from transformers.utils import logging
from typing import Tuple, List, Dict, Union, Optional

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "bond_dict.pkl"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "DaizeDong/GraphsGPT-1W": "https://huggingface.co/DaizeDong/GraphsGPT-1W/resolve/main/bond_dict.pkl",
        "DaizeDong/GraphsGPT-2W": "https://huggingface.co/DaizeDong/GraphsGPT-2W/resolve/main/bond_dict.pkl",
        "DaizeDong/GraphsGPT-4W": "https://huggingface.co/DaizeDong/GraphsGPT-4W/resolve/main/bond_dict.pkl",
        "DaizeDong/GraphsGPT-8W": "https://huggingface.co/DaizeDong/GraphsGPT-8W/resolve/main/bond_dict.pkl",
    },
    "tokenizer_file": {
        "DaizeDong/GraphsGPT-1W": "https://huggingface.co/DaizeDong/GraphsGPT-1W/resolve/main/tokenizer_config.json",
        "DaizeDong/GraphsGPT-2W": "https://huggingface.co/DaizeDong/GraphsGPT-2W/resolve/main/tokenizer_config.json",
        "DaizeDong/GraphsGPT-4W": "https://huggingface.co/DaizeDong/GraphsGPT-4W/resolve/main/tokenizer_config.json",
        "DaizeDong/GraphsGPT-8W": "https://huggingface.co/DaizeDong/GraphsGPT-8W/resolve/main/tokenizer_config.json",
    },
}

BOND_NUMBER_DOUBLE_MAP = {
    1.0: AllChem.BondType.SINGLE,
    1.5: AllChem.BondType.AROMATIC,
    2.0: AllChem.BondType.DOUBLE,
    3.0: AllChem.BondType.TRIPLE,
}


class GraphsGPTTokenizer(PreTrainedTokenizer):
    """
    Construct a GraphsGPT tokenizer.

    Args:
        vocab_file (`str`):
            Path to the bond dict.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "graph_position_ids_1", "graph_position_ids_2", "identifier_ids", "attention_mask"]

    def __init__(self, vocab_file: str, padding_id: int = 0, **kwargs):
        super().__init__(**kwargs)

        with open(vocab_file, "rb") as f:
            self.bond_dict = pickle.load(f)
        self.inverse_bond_dict = self._reverse_dict(self.bond_dict)

        self.padding_id = padding_id
        self.normalizer = MolStandardize.normalize.Normalizer()

        RDLogger.DisableLog('rdApp.*')

    def _reverse_dict(self, input_dict: Dict):
        output_dict = {}
        for key, value in input_dict.items():
            if value not in output_dict:
                output_dict[value] = key
            else:
                raise ValueError("Input dictionary does not satisfy the one-to-one mapping condition.")
        return output_dict

    @property
    def vocab_size(self):
        return len(self.bond_dict)

    def get_vocab(self):
        return {}

    def _sort_atom_idx(self, begin_atom_idx, end_atom_idx):
        if begin_atom_idx <= end_atom_idx:
            return begin_atom_idx, end_atom_idx
        else:
            return end_atom_idx, begin_atom_idx

    def _convert_smiles_to_standard_molecule(self, smiles: str) -> Optional[Chem.Mol]:
        mol = Chem.MolFromSmiles(smiles)

        if mol is not None:
            try:
                standardized_mol = self.normalizer.normalize(mol)
                return standardized_mol
            except:
                return None
        else:
            return None

    def _convert_smiles_or_molecule_to_standard_graph(self, smiles_or_mol: Union[str, Chem.Mol]) -> Union[Tuple[List[int], List[Tuple[int, int, float]]], None]:
        if isinstance(smiles_or_mol, str):
            mol = self._convert_smiles_to_standard_molecule(smiles_or_mol)
        elif isinstance(smiles_or_mol, Chem.Mol):
            mol = smiles_or_mol
        else:
            raise TypeError(f'"smiles_or_mol" must be either a str or a Chem.Mol (got {type(smiles_or_mol)})')

        if mol is not None:
            nodes = []  # atomic num of atoms
            edges = []  # begin and end atom ids, bond type

            for atom in mol.GetAtoms():
                nodes.append(atom.GetAtomicNum())  # [6, 6, 8]
            for bond in mol.GetBonds():
                edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble()))  # [(0, 1, 1.0), (1, 2, 1.0)]

            return nodes, edges
        else:
            return None

    def _convert_smiles_or_molecule_to_graph_sequence(self, smiles_or_mol: Union[str, Chem.Mol]) -> Union[Tuple[List[int], List[int], List[Tuple[int, int]]], None]:
        """convert SMILES to graph (nodes + edges) tokens"""
        result_graph = self._convert_smiles_or_molecule_to_standard_graph(smiles_or_mol)  # convert to graph

        if result_graph is not None:
            """convert graph (nodes + edges) to embedding ids"""
            atoms, bonds = result_graph

            id_list = []  # embedding index of nodes and edges (len: node_num + edge_num)
            mask_list = []  # mask the position of nodes (len: node_num + edge_num)
            connection_list = []  # index of connected nodes for all edges (len: edge_num)

            node_id_mapping = {}  # mapping relationship of the node index (id in "bonds" -> the order the nodes are added to "id_list")
            node_used_indicator = [False] * len(atoms)  # indicate whether the information of a specific node has been added to the "id_list"

            """convert according to the bond sequence"""
            for bond_info in bonds:
                begin_atom_id, end_atom_id, bond_type = bond_info
                begin_atom_id, end_atom_id = self._sort_atom_idx(begin_atom_id, end_atom_id)  # make sure "begin_atom_id" <= "end_atom_id"

                """get atomic num & bond_info"""
                begin_atom_atomic_num = atoms[begin_atom_id]
                end_atom_atomic_num = atoms[end_atom_id]

                min_atomic_num, max_atomic_num = self._sort_atom_idx(begin_atom_atomic_num, end_atom_atomic_num)
                bond_info_with_atomic_num = (min_atomic_num, max_atomic_num, bond_type)  # sort by atomic_num

                if bond_info_with_atomic_num not in self.bond_dict:
                    print(f"Encoding failed as the bond type \"{bond_info_with_atomic_num}\" is not in the bond dict.")
                    return None

                """add begin node"""
                if not node_used_indicator[begin_atom_id]:  # add only at the first time
                    id_list.append(begin_atom_atomic_num)  # directly use the atom's atomic num as its embedding id
                    mask_list.append(1)
                    node_id_mapping[begin_atom_id] = len(node_id_mapping)  # create node mapping to "id_list", used for "connection_list"
                    node_used_indicator[begin_atom_id] = True

                """add edge"""
                if not node_used_indicator[end_atom_id]:
                    node_id_mapping[end_atom_id] = len(node_id_mapping)  # create node mapping to "id_list", used for "connection_list"

                # add edge
                id_list.append(self.bond_dict[bond_info_with_atomic_num])  # id in "bond_dict"
                mask_list.append(0)
                connection_list.append((node_id_mapping[begin_atom_id], node_id_mapping[end_atom_id]))

                """add end node"""
                if not node_used_indicator[end_atom_id]:  # add only at the first time
                    id_list.append(end_atom_atomic_num)  # directly use the atom's atomic num as its embedding id
                    mask_list.append(1)
                    node_used_indicator[end_atom_id] = True

            return id_list, mask_list, connection_list

        else:
            return None

    def _convert_graph_sequence_to_list(self, id_list, mask_list, connection_list) -> Tuple[List[int], List[int], List[int], List[int]]:
        atom_and_bond_ids = id_list  # embedding ids for atom and bond features
        identifier_ids = mask_list  # embedding ids for node and edge identifiers
        graph_position_ids_1 = [0] * len(id_list)  # embedding ids for node and edge representation 1
        graph_position_ids_2 = [0] * len(id_list)  # embedding ids for node and edge representation 2

        # fill "graph_position_ids_1" and "graph_position_ids_2"
        now_node_id = 0
        now_edge_id = 0

        for i, identifier_id in enumerate(identifier_ids):
            if identifier_id == 1:  # is a node
                graph_position_ids_1[i] = now_node_id
                graph_position_ids_2[i] = now_node_id
                now_node_id += 1
            else:  # is an edge
                graph_position_ids_1[i] = connection_list[now_edge_id][0]
                graph_position_ids_2[i] = connection_list[now_edge_id][1]
                now_edge_id += 1

        return atom_and_bond_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids

    def _convert_graph_sequence_to_tensor(self, id_list, mask_list, connection_list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_and_bond_ids = torch.tensor(id_list, dtype=torch.int64)  # embedding ids for atom and bond features
        identifier_ids = torch.tensor(mask_list, dtype=torch.bool)  # embedding ids for node and edge identifiers
        graph_position_ids_1 = torch.zeros_like(atom_and_bond_ids, dtype=torch.int64)  # embedding ids for node and edge representation 1
        graph_position_ids_2 = torch.zeros_like(atom_and_bond_ids, dtype=torch.int64)  # embedding ids for node and edge representation 2

        # fill "graph_position_ids_1" and "graph_position_ids_2"
        node_num = int(torch.sum(identifier_ids).item())

        graph_position_ids_node = torch.arange(0, node_num, step=1, dtype=torch.int64)
        node_indices = torch.nonzero(identifier_ids)
        graph_position_ids_1[node_indices] = graph_position_ids_node.unsqueeze(1)
        graph_position_ids_2[node_indices] = graph_position_ids_node.unsqueeze(1)

        graph_position_ids_1_edge = []
        graph_position_ids_2_edge = []
        for (begin_node_id, end_node_id) in connection_list:
            graph_position_ids_1_edge.append(begin_node_id)
            graph_position_ids_2_edge.append(end_node_id)
        graph_position_ids_1_edge = torch.tensor(graph_position_ids_1_edge, dtype=torch.int64)
        graph_position_ids_2_edge = torch.tensor(graph_position_ids_2_edge, dtype=torch.int64)
        edge_indices = torch.nonzero(~identifier_ids)
        graph_position_ids_1[edge_indices] = graph_position_ids_1_edge.unsqueeze(1)
        graph_position_ids_2[edge_indices] = graph_position_ids_2_edge.unsqueeze(1)

        return atom_and_bond_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids

    def encode(self, smiles_or_mol: Union[str, Chem.Mol], return_tensors: str = None) -> Union[Dict[str, torch.LongTensor], Dict[str, List], None]:
        """
        Convert SMILES or molecule to tokens.
        Note that special representations like "[C@@H]" and "[N+]([O-])" will be lost.
        return: {
            "input_ids": torch.LongTensor or List
            "graph_position_ids_1": torch.LongTensor or List
            "graph_position_ids_2": torch.LongTensor or List
            "identifier_ids": torch.LongTensor or List
        }
        """
        graph_sequence = self._convert_smiles_or_molecule_to_graph_sequence(smiles_or_mol)

        if graph_sequence is not None:
            id_list, mask_list, connection_list = graph_sequence

            # Adjust all node & edge tokens to its true embedding indices in the model.
            # atom_and_bond_ids: atoms 1-118, bonds 119-?
            # input_ids: PAD 0, BOS 1, atoms 2-119, bonds 120-(?+1)
            if return_tensors is None:
                atom_and_bond_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids = self._convert_graph_sequence_to_list(id_list, mask_list, connection_list)
                input_ids = [id + 1 for id in atom_and_bond_ids]
            elif return_tensors == "pt":
                atom_and_bond_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids = self._convert_graph_sequence_to_tensor(id_list, mask_list, connection_list)
                input_ids = atom_and_bond_ids + 1  # 0 for padding tokens, 1 for BOS tokens
            else:
                raise NotImplementedError

            return {
                "input_ids": input_ids,
                "graph_position_ids_1": graph_position_ids_1,
                "graph_position_ids_2": graph_position_ids_2,
                "identifier_ids": identifier_ids,
            }

        else:
            print(f"\"{smiles_or_mol}\" cannot be converted to graph!")
            return None

    def encode_plus(self, smiles_or_mol: Union[str, Chem.Mol], return_tensors: str = None) -> Union[Dict[str, torch.LongTensor], Dict[str, List], None]:
        return self.encode(smiles_or_mol, return_tensors=return_tensors)

    def _pad_encoded_list(self, tokens_list: List[Dict[str, List]]) -> Dict[str, List]:
        # we use the right padding
        keys = tokens_list[0].keys()
        max_len = max(len(tokens["input_ids"]) for tokens in tokens_list)
        padded_list = {}

        for key in keys:
            padded_list[key] = [
                tokens[key] + [self.padding_id] * (max_len - len(tokens[key]))
                for tokens in tokens_list
            ]

        padded_list["attention_mask"] = [
            [1] * len(tokens["input_ids"]) + [0] * (max_len - len(tokens["input_ids"]))
            for tokens in tokens_list
        ]

        return padded_list

    def _pad_encoded_tensor(self, tokens_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # we use the right padding
        keys = tokens_list[0].keys()
        padded_tensors = {}

        for key in keys:
            tensors = [tensor_dict[key] for tensor_dict in tokens_list]
            padded_tensor = rnn_utils.pad_sequence(tensors, batch_first=True, padding_value=self.padding_id)
            padded_tensors[key] = padded_tensor

        padded_tensors["attention_mask"] = (padded_tensors["input_ids"] != self.padding_id)

        return padded_tensors

    def batch_encode(self, smiles_or_mol_list: List[Union[str, Chem.Mol]], return_tensors: str = None, nprocs: int = None) -> Union[Dict[str, torch.LongTensor], Dict[str, List[List]]]:
        """
        Convert SMILES list or molecule list to batched tokens.
        Note that special representations like "[C@@H]" and "[N+]([O-])" will be lost.
        return: {
            "input_ids": torch.LongTensor or List
            "graph_position_ids_1": torch.LongTensor or List
            "graph_position_ids_2": torch.LongTensor or List
            "identifier_ids": torch.LongTensor or List
            "attention_mask": torch.LongTensor or List
        }
        """
        if nprocs is None or nprocs <= 1:
            batched_tokens = []
            for smiles_or_mol in smiles_or_mol_list:
                result = self.encode(smiles_or_mol, return_tensors=return_tensors)
                if result is not None:
                    batched_tokens.append(result)

        else:  # If nprocs is specified, use Pool to process the encoding in parallel
            with Pool(nprocs) as p:
                batched_tokens = p.starmap(self.encode, [(smiles_or_mol, return_tensors) for smiles_or_mol in smiles_or_mol_list])
                batched_tokens = [token for token in batched_tokens if token is not None]

        # padding
        if return_tensors is None:
            batched_tokens = self._pad_encoded_list(batched_tokens)
        elif return_tensors == "pt":
            batched_tokens = self._pad_encoded_tensor(batched_tokens)
        else:
            raise NotImplementedError

        return batched_tokens

    def batch_encode_plus(self, smiles_or_mol_list: List[Union[str, Chem.Mol]], return_tensors: str = None, nprocs: int = None) -> Union[Dict[str, torch.LongTensor], Dict[str, List[List]]]:
        return self.batch_encode(smiles_or_mol_list, return_tensors=return_tensors, nprocs=nprocs)

    def _convert_token_lists_to_molecule(self, input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids, kekulize: bool = False) -> Optional[Chem.Mol]:
        try:
            # extract information
            new_input_ids = [id - 1 for id in input_ids]  # skip the padding and BOS

            atom_numbers = []
            bond_numbers = []
            bond_connections = []
            for i, identifier_id in enumerate(identifier_ids):
                if identifier_id == 0:  # bond
                    bond_id = new_input_ids[i]
                    bond_numbers.append(self.inverse_bond_dict[bond_id][2])
                    bond_connections.append((graph_position_ids_1[i], graph_position_ids_2[i]))
                else:  # atom
                    atom_numbers.append(new_input_ids[i])

            # create an empty mole
            editable_mol = Chem.EditableMol(Chem.Mol())

            # add atoms
            for atom_number in atom_numbers:
                atom = Chem.Atom(atom_number)
                editable_mol.AddAtom(atom)

            # add bonds
            for i in range(len(bond_numbers)):
                bond_number = bond_numbers[i]
                start_atom = bond_connections[i][0]
                end_atom = bond_connections[i][1]

                bond_type = BOND_NUMBER_DOUBLE_MAP[bond_number]
                editable_mol.AddBond(start_atom, end_atom, bond_type)

            mol = editable_mol.GetMol()
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    pass

                # convert from SMILES to avoid bugs
                smiles = self._convert_molecule_to_standard_smiles(mol)
                new_mol = self._convert_smiles_to_standard_molecule(smiles)
                if new_mol is not None:
                    mol = new_mol

        except:
            mol = None

        return mol

    def _convert_token_tensors_to_molecule(self, input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids, kekulize: bool = False) -> Optional[Chem.Mol]:
        try:
            # extract information
            seq_len = identifier_ids.shape[0]
            new_input_ids = input_ids - 1  # skip the padding and BOS

            atom_numbers = new_input_ids[identifier_ids].tolist()

            bond_ids = new_input_ids[~identifier_ids].tolist()
            bond_numbers = [self.inverse_bond_dict[bond_id][2] for bond_id in bond_ids]
            bond_connections = torch.stack((graph_position_ids_1, graph_position_ids_2), dim=0)[~identifier_ids.expand(2, seq_len)].reshape(2, -1).t().split(1, dim=0)
            bond_connections = [connection.tolist()[0] for connection in bond_connections]

            # create an empty mole
            editable_mol = Chem.EditableMol(Chem.Mol())

            # add atoms
            for atom_number in atom_numbers:
                atom = Chem.Atom(atom_number)
                editable_mol.AddAtom(atom)

            # add bonds
            for i in range(len(bond_numbers)):
                bond_number = bond_numbers[i]
                start_atom = bond_connections[i][0]
                end_atom = bond_connections[i][1]

                bond_type = BOND_NUMBER_DOUBLE_MAP[bond_number]
                editable_mol.AddBond(start_atom, end_atom, bond_type)

            mol = editable_mol.GetMol()
            if kekulize:
                try:
                    Chem.Kekulize(mol, clearAromaticFlags=True)
                except:
                    pass

                # convert from SMILES to avoid bugs
                smiles = self._convert_molecule_to_standard_smiles(mol)
                new_mol = self._convert_smiles_to_standard_molecule(smiles)
                if new_mol is not None:
                    mol = new_mol

        except:
            mol = None

        return mol

    def _convert_molecule_to_standard_smiles(self, mol: Chem.Mol) -> str:
        if mol is not None:
            try:
                standardized_mol = self.normalizer.normalize(mol)
            except:
                standardized_mol = mol
            canonical_smiles = Chem.MolToSmiles(standardized_mol)  # isomericSmiles=True

        else:
            canonical_smiles = None

        return canonical_smiles

    def _decode_single_list(self, single_input_ids, single_graph_position_ids_1, single_graph_position_ids_2, single_identifier_ids, kekulize=False) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        mol = self._convert_token_lists_to_molecule(single_input_ids, single_graph_position_ids_1, single_graph_position_ids_2, single_identifier_ids, kekulize=kekulize)
        smiles = self._convert_molecule_to_standard_smiles(mol)
        return mol, smiles

    def _decode_single_tensor(self, single_input_ids, single_graph_position_ids_1, single_graph_position_ids_2, single_identifier_ids, kekulize=False) -> Tuple[Optional[Chem.Mol], Optional[str]]:
        mol = self._convert_token_tensors_to_molecule(single_input_ids, single_graph_position_ids_1, single_graph_position_ids_2, single_identifier_ids, kekulize=kekulize)
        smiles = self._convert_molecule_to_standard_smiles(mol)
        return mol, smiles

    def decode(self, tokens: Union[Dict[str, torch.LongTensor], Dict[str, List]], kekulize: bool = False, nprocs: int = None) -> Union[Tuple[Chem.Mol, str], Tuple[List[Chem.Mol], List[str]]]:
        """
        Convert encoded tokens to the molecule and SMILES.
        Note that special representations like "[C@@H]" and "[N+]([O-])" will be lost.
        tokens: dict of tensors or dict of lists
        return:
        (
            molecule,
            SMILES
        )
        or
        (
            [molecule, molecule, ..., molecule],
            [SMILES, SMILES, ..., SMILES]
        )
        """

        for key in ("input_ids", "graph_position_ids_1", "graph_position_ids_2", "identifier_ids"):
            if key not in tokens:
                raise KeyError(f'The tokens dictionary must contain the key "{key}"!')

        input_ids = tokens["input_ids"]
        graph_position_ids_1 = tokens["graph_position_ids_1"]
        graph_position_ids_2 = tokens["graph_position_ids_2"]
        identifier_ids = tokens["identifier_ids"]
        attention_mask = tokens["attention_mask"] if "attention_mask" in tokens else None

        if isinstance(input_ids, list):
            if isinstance(input_ids[0], list):  # input is a batched sample
                if attention_mask is not None:
                    for i in range(len(input_ids)):
                        this_non_padding_len = sum(attention_mask[i])
                        input_ids[i] = input_ids[i][:this_non_padding_len]
                        graph_position_ids_1[i] = graph_position_ids_1[i][:this_non_padding_len]
                        graph_position_ids_2[i] = graph_position_ids_2[i][:this_non_padding_len]
                        identifier_ids[i] = identifier_ids[i][:this_non_padding_len]

                if nprocs is None or nprocs <= 1:
                    mol_list = []
                    smiles_list = []
                    for i in range(len(input_ids)):
                        mol, smiles = self._decode_single_list(
                            input_ids[i],
                            graph_position_ids_1[i],
                            graph_position_ids_2[i],
                            identifier_ids[i],
                            kekulize=kekulize,
                        )
                        mol_list.append(mol)
                        smiles_list.append(smiles)

                else:  # If nprocs is specified, use Pool to process the decoding in parallel
                    with Pool(nprocs) as p:
                        results = p.starmap(
                            self._decode_single_list,
                            [(input_ids[i], graph_position_ids_1[i], graph_position_ids_2[i], identifier_ids[i], kekulize)
                             for i in range(len(input_ids))]
                        )
                        mol_list = [result[0] for result in results]
                        smiles_list = [result[1] for result in results]

                return mol_list, smiles_list

            else:  # input is a single sample
                if attention_mask is not None:
                    non_padding_len = sum(attention_mask)  # different here
                    input_ids = input_ids[:non_padding_len]
                    graph_position_ids_1 = graph_position_ids_1[:non_padding_len]
                    graph_position_ids_2 = graph_position_ids_2[:non_padding_len]
                    identifier_ids = identifier_ids[:non_padding_len]

                mol, smiles = self._decode_single_list(input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids, kekulize=kekulize)

                return mol, smiles

        elif isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 2:  # input is a batched sample
                input_ids = list(torch.split(input_ids, 1, dim=0))
                graph_position_ids_1 = list(torch.split(graph_position_ids_1, 1, dim=0))
                graph_position_ids_2 = list(torch.split(graph_position_ids_2, 1, dim=0))
                identifier_ids = list(torch.split(identifier_ids, 1, dim=0))

                if attention_mask is not None:
                    for i in range(len(input_ids)):
                        this_non_padding_len = torch.sum(attention_mask[i]).item()
                        input_ids[i] = input_ids[i][:, :this_non_padding_len]
                        graph_position_ids_1[i] = graph_position_ids_1[i][:, :this_non_padding_len]
                        graph_position_ids_2[i] = graph_position_ids_2[i][:, :this_non_padding_len]
                        identifier_ids[i] = identifier_ids[i][:, :this_non_padding_len]

                if nprocs is None or nprocs <= 1:
                    mol_list = []
                    smiles_list = []
                    for i in range(len(input_ids)):
                        mol, smiles = self._decode_single_tensor(
                            input_ids[i].squeeze(0),
                            graph_position_ids_1[i].squeeze(0),
                            graph_position_ids_2[i].squeeze(0),
                            identifier_ids[i].squeeze(0),
                            kekulize=kekulize,
                        )
                        mol_list.append(mol)
                        smiles_list.append(smiles)

                else:  # If nprocs is specified, use Pool to process the decoding in parallel
                    # tensors must be on CPU to call multiprocessing
                    with Pool(nprocs) as p:
                        results = p.starmap(
                            self._decode_single_tensor,
                            [
                                (
                                    input_ids[i].squeeze(0).clone().cpu(),
                                    graph_position_ids_1[i].squeeze(0).clone().cpu(),
                                    graph_position_ids_2[i].squeeze(0).clone().cpu(),
                                    identifier_ids[i].squeeze(0).clone().cpu(),
                                    kekulize
                                )
                                for i in range(len(input_ids))
                            ]
                        )
                        mol_list = [result[0] for result in results]
                        smiles_list = [result[1] for result in results]

                return mol_list, smiles_list

            else:  # input is a single sample
                if attention_mask is not None:
                    non_padding_len = torch.sum(attention_mask)  # different here
                    input_ids = input_ids[:non_padding_len]
                    graph_position_ids_1 = graph_position_ids_1[:non_padding_len]
                    graph_position_ids_2 = graph_position_ids_2[:non_padding_len]
                    identifier_ids = identifier_ids[:non_padding_len]

                mol, smiles = self._decode_single_tensor(input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids, kekulize=kekulize)

                return mol, smiles

        else:
            raise TypeError("The tokens[\"input_ids\"] must be of type torch.Tensor or list!")

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save only the vocabulary of the tokenizer (vocabulary + added tokens).

        This method won't save the configuration and special token mappings of the tokenizer. Use
        [`~PreTrainedTokenizerFast._save_pretrained`] to save the whole state of the tokenizer.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.
            filename_prefix (`str`, *optional*):
                An optional prefix to add to the named of the saved files.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return

        bond_dict_file = os.path.join(save_directory, "bond_dict.pkl")
        with open(bond_dict_file, "wb") as f:
            pickle.dump(self.bond_dict, f)

        return (bond_dict_file,)


if __name__ == "__main__":
    smiles_list = [
        "CCCNC(=O)Cc1csc(NC(=O)NCC2([C@@H](O)C(C)C)CC2)n1",
        "O=[N+]([O-])c1c(NCCCn2ccnc2)nc2ccccn12",
        "CNC(=O)c1cccc(C(=O)N2C[C@H]3CCN(Cc4n[nH]c(C)n4)C[C@H]32)c1",
        "COC(C)(C)c1nsc(NC(=O)[C@H](C)c2cnccn2)n1",
        "COc1cc(OC)c(Cn2nccc2NC(=O)c2ccc(N(C)C)cn2)c(OC)c1",
        "Cn1cc(C(=O)N[C@H]2CC[C@@H](CNCc3ccc(F)cc3)C2)nc1O",
        "O=C(CCc1ccc(Cl)cc1)Nc1ccc2c(c1)S(=O)(=O)CC2",
        "CS[C@H](CNC(=O)[C@@H](C)Oc1ccc(C#N)cc1)C(C)(C)C",
        "CC(=O)N1CCc2cc(C(=O)[C@H](C)OC(=O)c3cnn(C4CCCCC4)c3C)ccc21",
        "CC[C@H](C)[C@@H](NC(=O)Cc1cc(C)no1)C(=O)NCCN1CCOCC1",
    ]

    tokenizer = GraphsGPTTokenizer.from_pretrained("DaizeDong/GraphsGPT-1W")

    """encode a batched SMILES"""
    # inputs = tokenizer.batch_encode(smiles_list, nprocs=4)  # list
    inputs = tokenizer.batch_encode(smiles_list, return_tensors="pt", nprocs=4)  # tensor
    decoded_mol_list, decoded_smiles_list = tokenizer.decode(inputs, nprocs=4)
    print("inputs:", inputs, "\n")
    print("decoded_smiles_list:", decoded_smiles_list, "\n")

    # decoded_smiles_list: [
    #     'CCCNC(=O)Cc1csc(NC(=O)NCC2(C(O)C(C)C)CC2)n1',
    #     'O=N(O)c1c(NCCCn2ccnc2)nc2ccccn12',
    #     'CNC(=O)c1cccc(C(=O)N2CC3CCN(Cc4nnc(C)n4)CC32)c1',
    #     'COC(C)(C)c1nsc(NC(=O)C(C)c2cnccn2)n1',
    #     'COc1cc(OC)c(Cn2nccc2NC(=O)c2ccc(N(C)C)cn2)c(OC)c1',
    #     'Cn1cc(C(=O)NC2CCC(CNCc3ccc(F)cc3)C2)nc1O',
    #     'O=C(CCc1ccc(Cl)cc1)Nc1ccc2c(c1)S(=O)(=O)CC2',
    #     'CSC(CNC(=O)C(C)Oc1ccc(C#N)cc1)C(C)(C)C',
    #     'CC(=O)N1CCc2cc(C(=O)C(C)OC(=O)c3cnn(C4CCCCC4)c3C)ccc21',
    #     'CCC(C)C(NC(=O)Cc1cc(C)no1)C(=O)NCCN1CCOCC1'
    # ]

    """encode a single SMILES"""
    # inputs = tokenizer.encode(smiles_list[0])  # list
    inputs = tokenizer.encode(smiles_list[0], return_tensors="pt")  # tensor
    decoded_mol, decoded_smiles = tokenizer.decode(inputs)
    print("inputs:", inputs, "\n")
    print("decoded_smiles:", decoded_smiles, "\n")

    # decoded_smiles: 'CCCNC(=O)Cc1csc(NC(=O)NCC2(C(O)C(C)C)CC2)n1'
