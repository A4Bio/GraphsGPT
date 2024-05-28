import numpy as np
import torch
from typing import Tuple

from utils.algorithms import FindCycles

VALENCE_LIMIT = {
    # ATOMIC_NUM: MAX_VALENCE
    5: 3,  # B
    6: 4,  # C
    7: 3,  # N
    8: 2,  # 0
    9: 1,  # F
    14: 4,  # Si
    16: 6,  # S
    17: 1,  # Cl
    35: 1,  # Br
    53: 1,  # I
}


def get_atom_ids_from_bond_id(inverse_bond_dict, bond_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    atom_ids = inverse_bond_dict[bond_id.item()][:2]
    return (
        torch.tensor(atom_ids[0], dtype=bond_id.dtype, device=bond_id.device).unsqueeze(0),
        torch.tensor(atom_ids[1], dtype=bond_id.dtype, device=bond_id.device).unsqueeze(0)
    )


def get_another_atom_id_from_existing_bond(inverse_bond_dict, bond_id: torch.Tensor, connected_atom_id_1: torch.Tensor) -> torch.Tensor:
    # bond_id: id in the bond_dict
    # connected_atom_id_1: atomic_num of the existing connected atom
    info = inverse_bond_dict[bond_id.item()]  # (min_atomic_num, max_atomic_num, bond_type)
    if not (connected_atom_id_1.item() == info[0] or connected_atom_id_1.item() == info[1]):
        raise ValueError
    elif connected_atom_id_1.item() == info[0]:
        return torch.tensor(info[1], dtype=bond_id.dtype, device=bond_id.device).unsqueeze(0)
    else:
        return torch.tensor(info[0], dtype=bond_id.dtype, device=bond_id.device).unsqueeze(0)


def check_bond_connectivity_begin(inverse_bond_dict, bond_id, atom_id):
    # bond_id: id in the bond_dict
    # atom_id: atomic_num
    info = inverse_bond_dict[bond_id]  # (min_atomic_num, max_atomic_num, bond_type)
    if atom_id == info[0] or atom_id == info[1]:  # "+1" for getting the atomic num
        return True
    else:
        return False


def check_bond_connectivity_both_sides(inverse_bond_dict, bond_id, atom_id1, atom_id2):
    # bond_id: id in the bond_dict
    # atom_id: atomic_num
    if atom_id1 <= atom_id2:
        min_atomic_num = atom_id1
        max_atomic_num = atom_id2
    else:
        min_atomic_num = atom_id2
        max_atomic_num = atom_id1

    info = inverse_bond_dict[bond_id]  # (min_atomic_num, max_atomic_num, bond_type)
    if min_atomic_num == info[0] and max_atomic_num == info[1]:
        return True
    else:
        return False


def check_bond_in_graph(graph_position_ids_1, graph_position_ids_2, connection_1, connection_2):
    connection_1_in_1 = (graph_position_ids_1 == connection_1)
    connection_1_in_2 = (graph_position_ids_2 == connection_1)
    connection_2_in_1 = (graph_position_ids_1 == connection_2)
    connection_2_in_2 = (graph_position_ids_2 == connection_2)
    if torch.any(connection_1_in_1 & connection_2_in_2) or torch.any(connection_1_in_2 & connection_2_in_1):
        return True
    else:
        return False


def get_valence(input_ids, graph_position_ids_1, graph_position_ids_2, connection_id, inverse_bond_dict, bond_mask):
    """
    Return the total valence for an atom by checking its connected bonds
    For each connected aromatic bond with 1.5 valence, the real valence for the atom can be adjusted by a value of ±0.5
    """
    bond_ids_in_dict = input_ids[bond_mask] - 1
    bond_connections_1 = graph_position_ids_1[bond_mask]
    bond_connections_2 = graph_position_ids_2[bond_mask]
    bond_connections_mask = (bond_connections_1 == connection_id) | (bond_connections_2 == connection_id)
    connected_bond_ids = bond_ids_in_dict[bond_connections_mask].tolist()

    total_valence = 0
    valence_offset = 0
    for bond_type in connected_bond_ids:
        this_valence = inverse_bond_dict[bond_type][2]
        total_valence += this_valence
        if this_valence == 1.5:  # aromatic bond
            valence_offset += 0.5

    if int(valence_offset / 0.5) % 2 == 0:
        total_valence += int(valence_offset / 0.5) % 2
        valence_offset = 0
    else:
        valence_offset = 0.5

    return total_valence, valence_offset


def fix_dissociative_aromatic_bond(input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids, inverse_bond_dict, bond_dict):
    """
    Replace invalid aromatic bonds with corresponding single/double/triple bonds.
    Invalid aromatic bonds: non-ring & incomplete ring
    """
    atom_num = torch.sum(identifier_ids).item()
    bond_mask = ~identifier_ids

    indices_all_positions = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.int64)
    indices_bonds = indices_all_positions[bond_mask]

    # Create connection -> bond_type mapping
    bond_ids_in_dict = (input_ids.clone()[bond_mask] - 1).tolist()

    bond_connections_1 = graph_position_ids_1[bond_mask].tolist()
    bond_connections_2 = graph_position_ids_2[bond_mask].tolist()
    bond_connections_all = [
        (bond_connections_1[i], bond_connections_2[i]) if bond_connections_1[i] <= bond_connections_2[i] else (bond_connections_2[i], bond_connections_1[i])  # sort node id
        for i in range(len(bond_connections_1))
    ]

    connection_bond_mapping = {}
    connection_index_mapping = {}  # record the positions for each connection

    for i, connection in enumerate(bond_connections_all):
        connection_bond_mapping[connection] = inverse_bond_dict[bond_ids_in_dict[i]]
        connection_index_mapping[connection] = indices_bonds[i]

    # Find cycles in the molecule
    adjacency_matrix = np.zeros((atom_num, atom_num), dtype=np.int8)
    for connection in bond_connections_all:
        adjacency_matrix[connection[0], connection[1]] = 1
        adjacency_matrix[connection[1], connection[0]] = 1
    cycle_finder = FindCycles(adjacency_matrix)
    cycles = cycle_finder.find_cycles()

    # Check the bonds in each cycle
    # If there is any cycle with all aromatic bonds, then all bonds in it are marked as valid
    valid_aromatic_connections = set()

    for cycle in cycles:
        is_aromatic = []
        not_aromatic = []

        # Check the validity
        for node_id in range(len(cycle)):
            if node_id < len(cycle) - 1:
                connection = (cycle[node_id], cycle[node_id + 1]) if cycle[node_id] <= cycle[node_id + 1] else (cycle[node_id + 1], cycle[node_id])
            else:
                connection = (cycle[node_id], cycle[0]) if cycle[node_id] <= cycle[0] else (cycle[0], cycle[node_id])

            begin_atomic_num, end_atomic_num, bond_type = connection_bond_mapping[connection]

            if bond_type == 1.5:
                is_aromatic.append(connection)
            else:
                not_aromatic.append(connection)

        if len(not_aromatic) == 0:  # all bonds are aromatic
            for connection in is_aromatic:
                valid_aromatic_connections.add(connection)

    # Change invalid aromatic bonds into single/double/triple bonds
    for connection in bond_connections_all:
        begin_atomic_num, end_atomic_num, bond_type = connection_bond_mapping[connection]
        if bond_type == 1.5 and connection not in valid_aromatic_connections:  # invalid aromatic
            index = connection_index_mapping[connection]
            if (begin_atomic_num, end_atomic_num, 1.0) in bond_dict:  # single
                new_bond_id = bond_dict[(begin_atomic_num, end_atomic_num, 1.0)] + 1
            elif (begin_atomic_num, end_atomic_num, 2.0) in bond_dict:  # double
                new_bond_id = bond_dict[(begin_atomic_num, end_atomic_num, 2.0)] + 1
            elif (begin_atomic_num, end_atomic_num, 3.0) in bond_dict:  # triple
                new_bond_id = bond_dict[(begin_atomic_num, end_atomic_num, 3.0)] + 1
            else:  # this bond is incorrigible!
                continue
            input_ids[index] = new_bond_id

    return input_ids, graph_position_ids_1, graph_position_ids_2, identifier_ids
