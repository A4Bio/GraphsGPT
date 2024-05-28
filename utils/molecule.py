from rdkit.Chem import MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_molecule_standard_scaffold(mol, normalizer=None):
    if normalizer is None:
        normalizer = MolStandardize.normalize.Normalizer()

    if mol is not None:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)

            try:
                standardized_scaffold = normalizer.normalize(scaffold)
            except:
                standardized_scaffold = scaffold

            return standardized_scaffold
        except:
            return None
    else:
        return None
