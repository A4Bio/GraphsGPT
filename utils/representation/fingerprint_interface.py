import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from data.tokenizer import GraphsGPTTokenizer
from models.graphsgpt.modeling_graphsgpt import GraphsGPTForCausalLM

CLASS_OF_FPGEN = {
    "fingerprint-morgan": rdFingerprintGenerator.GetMorganGenerator,
    "fingerprint-rdkit": rdFingerprintGenerator.GetRDKitFPGenerator,
    "fingerprint-ap": rdFingerprintGenerator.GetAtomPairGenerator,
    "fingerprint-tt": rdFingerprintGenerator.GetTopologicalTorsionGenerator,
}


class FPModel(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        if model_name == 'graphsgpt':
            self.model = GraphsGPTForCausalLM.from_pretrained("DaizeDong/GraphsGPT-1W")
            self.tokenizer = GraphsGPTTokenizer.from_pretrained("DaizeDong/GraphsGPT-1W")
            self.model.cuda()
            self.model.eval()

        if model_name == 'morgan':
            self.radius = 2
            self.fpsize = 2048
            self.count = 0

        self.memory = {}

    def gen_fp(self, mol, fpgen, count=False):
        if count:
            g = fpgen.GetCountFingerprintAsNumPy
        else:
            g = fpgen.GetFingerprintAsNumPy
        return np.array(g(mol))

    def forward(self, smiles_list):
        if all([one in self.memory for one in smiles_list]):
            fingerprint_tokens = torch.stack([self.memory[one] for one in smiles_list], dim=0)
            return fingerprint_tokens

        if self.model_name == 'graphsgpt':
            with torch.no_grad():
                batch = self.tokenizer.batch_encode(smiles_list, return_tensors="pt")
                inputs = batch['batched_tokens']
                inputs = {k: v.cuda() for k, v in inputs.items()}
                fingerprint_tokens = self.model.encode_to_fingerprints(**inputs)
                fingerprint_tokens = fingerprint_tokens[:, 0].cpu()

        if self.model_name == 'morgan':
            fps = []
            fpgen = CLASS_OF_FPGEN["fingerprint-morgan"](radius=self.radius, fpSize=self.fpsize)
            for i in range(len(smiles_list)):
                mol = Chem.MolFromSmiles(smiles_list[i])
                fps.append(self.gen_fp(mol, fpgen, count=self.count))
            fingerprint_embeddings = np.array(fps).astype(float)

            fingerprint_tokens = torch.from_numpy(fingerprint_embeddings)

        for i, one in enumerate(smiles_list):
            self.memory[one] = fingerprint_tokens[i]

        return fingerprint_tokens
