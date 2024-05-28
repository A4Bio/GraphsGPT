# Conditional Generation with GraphGPT-C Decoder

The conditional generation of GraphGPT-C is similar to the unconditional version. However, there are some extra configurations to control the properties. For other configurations you can refer to [Unconditional Generation](..%2Funconditional%2FREADME-Generation-Uncond.md).



## Extra Generation Configurations

- **Conditions:**
    - `value_qed`: `(float) None` (The target QED value for generated molecules. The model will not condition on this property if not specified.)
    - `value_sa`: `(float) None` (The target SA score for generated molecules. The model will not condition on this property if not specified.)
    - `value_logp`: `(float) None` (The target logP value for generated molecules. The model will not condition on this property if not specified.)
    - `scaffold_smiles`: `(str) None` (The target scaffold SMILES. The model will not condition on this property if not specified.)



### Configurations in the Paper

We use the following configuration to test the ability of GraphGPT on conditioning molecular properties:

````bash
strict_generation="False"
fix_aromatic_bond="True"

do_sample="False"

check_first_node="True"
check_atom_valence="True"
````

Example scripts can be found in `scripts/generation/unconditional/examples`.



### Generate with More Diversity

You can further turn on the *probabilistic sampling* for more diversity:

````bash
strict_generation="False"
fix_aromatic_bond="True"

do_sample="True"
top_k=4
temperature=1.0

check_first_node="True"
check_atom_valence="True"
````



## Evaluate & Visualize the Results

Run `scripts/generation/conditional/visualize.sh`.

The mean and variance property values of generated molecules will also be saved to the `summary.txt`.



