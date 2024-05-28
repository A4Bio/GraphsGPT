import numpy as np
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, Mol
from rdkit.Chem.QED import qed

from utils.property_scores.sascorer import compute_sa_score
from utils.property_scores.tanimoto_similarity import MolsTanimotoSimilarity


def fix_explicit_hs(mol: Mol) -> Mol:
    # rdkit has a problem with implicit hs. By default there are only explicit hs.
    # This is a hack to fix this error
    for a in mol.GetAtoms():
        a.SetNoImplicit(False)

    mol = Chem.AddHs(mol, explicitOnly=True)
    mol = Chem.RemoveHs(mol)

    Chem.SanitizeMol(mol)
    return mol


def get_basic(mol):
    n_atoms = len(mol.GetAtoms())
    n_bonds = len(mol.GetBonds())
    n_rings = len(Chem.GetSymmSSSR(mol))
    weight = Descriptors.ExactMolWt(mol)
    return n_atoms, n_bonds, n_rings, weight


def get_is_valid(mol):
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return False
    return True


def get_qed(mol):
    # mol = fix_explicit_hs(mol)
    # return qed(mol)
    try:
        mol = deepcopy(mol)
        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        return qed(mol)
    except Exception as e:
        print(e)
        return None


def get_sa(mol):
    try:
        mol = deepcopy(mol)
        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        return compute_sa_score(mol)
    except Exception as e:
        print(e)
        return None


def get_logp(mol):
    try:
        mol = deepcopy(mol)
        try:
            AllChem.Kekulize(mol, clearAromaticFlags=True)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        return Crippen.MolLogP(mol)
    except Exception as e:
        print(e)
        return None


def get_lipinski(mol):
    try:
        mol = deepcopy(mol)
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        mol.UpdatePropertyCache(strict=False)
        rule_1 = Descriptors.ExactMolWt(mol) < 500
        rule_2 = Lipinski.NumHDonors(mol) <= 5
        rule_3 = Lipinski.NumHAcceptors(mol) <= 10
        logp = Crippen.MolLogP(mol)
        rule_4 = (logp >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])
    except Exception as e:
        print(e)
        return None


def get_hacc(mol):
    return Lipinski.NumHAcceptors(mol)


def get_hdon(mol):
    return Lipinski.NumHDonors(mol)


def get_rdkit_rmsd(mol, n_conf=20, random_seed=42):
    """
    Calculate the alignment of generated mol and rdkit predicted mol
    Return the rmsd (max, min, median) of the `n_conf` rdkit conformers
    """
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    mol3d = Chem.AddHs(mol)
    rmsd_list = []
    # predict 3d
    confIds = AllChem.EmbedMultipleConfs(mol3d, n_conf, randomSeed=random_seed)
    for confId in confIds:
        AllChem.UFFOptimizeMolecule(mol3d, confId=confId)
        rmsd = Chem.rdMolAlign.GetBestRMS(mol, mol3d, refId=confId)
        rmsd_list.append(rmsd)
    # mol3d = Chem.RemoveHs(mol3d)
    rmsd_list = np.array(rmsd_list)
    return [np.max(rmsd_list), np.min(rmsd_list), np.median(rmsd_list)]


def get_tanimoto_similarity(mol_list):
    tanimoto_calculator = MolsTanimotoSimilarity(mol_list)
    similarity_matrix = tanimoto_calculator.get_tanimoto_similarity_matrix()
    top_similarities = tanimoto_calculator.get_top_tanimoto_similarities()

    return similarity_matrix, top_similarities
