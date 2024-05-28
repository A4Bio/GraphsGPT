import numpy as np
from rdkit import DataStructs
from rdkit.Chem import AllChem


class MolsTanimotoSimilarity:
    def __init__(self, mols):
        self.mols = mols
        self.fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]  # get fingerprints
        self.tanimoto_similarity_matrix = np.zeros((len(mols), len(mols)))

    def get_tanimoto_similarity_matrix(self):
        # calculate similarity for each mol pair
        for i in range(0, len(self.mols)):
            for j in range(i + 1, len(self.mols)):
                similarity = DataStructs.TanimotoSimilarity(self.fingerprints[i], self.fingerprints[j])
                self.tanimoto_similarity_matrix[i, j] = similarity

        # complete the similarity matrix
        lower_indices = np.tril_indices(len(self.mols), -1)
        self.tanimoto_similarity_matrix[lower_indices] = self.tanimoto_similarity_matrix.T[lower_indices]
        return self.tanimoto_similarity_matrix

    def get_top_tanimoto_similarities(self):
        top_similarities = np.max(self.tanimoto_similarity_matrix, axis=1)
        return top_similarities.tolist()
