# Unconditional Generation with GraphGPT Decoder

The generation of GraphsGPT is controlled by multiple adjustable configurations. You can refer to the following descriptions to adjust the generation to different needs.

## Generation Configurations

- **Auto-fix Toggle:**
    - `strict_generation`: `(bool) True` (Whether to tolerate the validity exceptions during generation. Setting to `False` will enable flexible generation that automatically fixes invalid predictions and ensure maximum effectiveness.)
    - `fix_aromatic_bond`: `(bool) False` (Whether to fix the dissociative aromatic bonds in the generated molecules.)


- **Sampling Strategy:**
    - `do_sample`: `(bool) False` (Whether to use probabilistic sampling for bond predictions. Setting to `True` will enable probabilistic sampling and introduce more randomness.)
    - `top_k`: `(int) None` (The range of top predictions for probability sampling. Available when `do_sample` is `True`.)
    - `temperature`: `(float) 1.0` (Temperature to adjust the probability distribution. Available when `do_sample` is `True`.)


- **Hyperparameters:**
    - `max_atoms`: `(int) None` (The maximum number of atoms for generation.)
    - `similarity_threshold`: `(float) 0.5` (Threshold for classifying whether a generated atom is new or old.)


- **Other Check Terms:**
    - `check_first_node`: `(bool) True` (Whether to check the consistency between the predicted beginning atom and the first bond, and fix the order of the beginning two atoms.)
    - `check_atom_valence`: `(bool) False` (Whether to check the validity regarding the valence of the atoms connected to the predicted bonds.)

For reference, we provide some example configurations to use under different circumstances.

### Validate the Pretraining Performance

To validate the pretraining performance, the generation should be of no randomness, both *auto-fix* and *probabilistic sampling* should be turned off:

````bash
strict_generation="True"
fix_aromatic_bond="False"

do_sample="False"

check_first_node="True"
check_atom_valence="False"
````

An example script can be found in `scripts/generation/unconditional/examples/generate_strict.sh`.

### Generate with More Effectiveness

Upon generation of more effective results. You can turn on the *auto-fix*:

````bash
strict_generation="False"
fix_aromatic_bond="True"

do_sample="False"

check_first_node="True"
check_atom_valence="False"
````

An example script can be found in `scripts/generation/unconditional/examples/generate_flexible.sh`.

### Generate with More Diversity (Further Finetuning Needed)

To generate more diverse results. You can further turn on the *probabilistic sampling* and *valence check*. This requires the encoded Graph Words $\mathcal{W}$ to be of variable information, where an extra finetuning would be needed.

````bash
strict_generation="False"
fix_aromatic_bond="True"

do_sample="True"
top_k=4
temperature=1.0

check_first_node="True"
check_atom_valence="True"
````

## Visualize the Results

Run `scripts/generation/unconditional/visualize.sh`.

