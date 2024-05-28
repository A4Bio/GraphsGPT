import multiprocessing

import networkx as nx
import numpy as np
from eden.graph import vectorize
from rdkit import Chem
from sklearn.metrics.pairwise import pairwise_kernels
from tqdm import tqdm


def single_mol_to_nx(mol):
    if mol is not None:
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), label=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), label=int(bond.GetBondTypeAsDouble()))
        return G
    return None


def mols_to_nx(mols, n_jobs=None):
    # convert with multiprocessing support
    if n_jobs is not None:
        pool = multiprocessing.Pool(processes=n_jobs)
        nx_graphs = pool.map(single_mol_to_nx, mols)
        pool.close()
        pool.join()
        return [graph for graph in tqdm(nx_graphs, desc='Converting Molecules to Graphs') if graph is not None]
    else:
        nx_graphs = [single_mol_to_nx(mol) for mol in tqdm(mols, desc='Converting Molecules to Graphs')]
        return [graph for graph in nx_graphs if graph is not None]


def single_graph_to_vector(graph):
    return vectorize(graph, complexity=4, discrete=True).toarray()


def compute_nspdk_mmd(samples1, samples2, metric='linear', n_jobs=None):
    # code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
    # convert with multiprocessing support
    if n_jobs is not None:
        pool = multiprocessing.Pool(processes=n_jobs)
        vectors1 = pool.map(single_graph_to_vector, [[sample] for sample in samples1])
        vectors2 = pool.map(single_graph_to_vector, [[sample] for sample in samples2])
        pool.close()
        pool.join()
        vectors1 = np.concatenate([vector for vector in tqdm(vectors1, desc='Vectorization...')], axis=0)
        vectors2 = np.concatenate([vector for vector in tqdm(vectors2, desc='Vectorization...')], axis=0)
    else:
        print("Vectorization...")
        vectors1 = vectorize(samples1, complexity=4, discrete=True).toarray()
        vectors2 = vectorize(samples2, complexity=4, discrete=True).toarray()

    print("Computing X...")
    X = pairwise_kernels(vectors1, None, metric=metric, n_jobs=n_jobs)
    print(f"X={X}")

    print("Computing Y...")
    Y = pairwise_kernels(vectors2, None, metric=metric, n_jobs=n_jobs)
    print(f"Y={Y}")

    print("Computing Z...")
    Z = pairwise_kernels(vectors1, vectors2, metric=metric, n_jobs=n_jobs)
    print(f"Z={Z}")

    return np.average(X) + np.average(Y) - 2 * np.average(Z)


def get_npsdk(smiles_list1, smiles_list2, metric='linear', n_jobs=None):
    nx_graphs1 = mols_to_nx([Chem.MolFromSmiles(smile) for smile in tqdm(smiles_list1, desc='Converting SMILES to Molecules')], n_jobs=n_jobs)
    nx_graphs2 = mols_to_nx([Chem.MolFromSmiles(smile) for smile in tqdm(smiles_list2, desc='Converting SMILES to Molecules')], n_jobs=n_jobs)
    nx_graphs1_remove_empty = [G for G in nx_graphs1 if not G.number_of_nodes() == 0]
    nx_graphs2_remove_empty = [G for G in nx_graphs2 if not G.number_of_nodes() == 0]
    nspdk_mmd = compute_nspdk_mmd(nx_graphs1_remove_empty, nx_graphs2_remove_empty, metric=metric, n_jobs=n_jobs)
    return nspdk_mmd
