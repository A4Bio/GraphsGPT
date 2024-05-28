import os
import pickle
import torch
from argparse import ArgumentParser
from rdkit import Chem

from data.tokenizer import GraphsGPTTokenizer
from models.graphsgpt_cond_gen.modelling_graphsgpt_cond_gen import GraphsGPTForConditionalGeneration
from utils.io import create_dir, load_json
from utils.molecule import get_molecule_standard_scaffold
from utils.operations.operation_list import split_list_with_yield
from utils.operations.operation_number import normalize_value
from utils.operations.operation_string import str2bool
from utils.operations.operation_tensor import move_tensors_to_device
from utils.property_scores.scoring_func import get_qed, get_sa, get_logp


def get_input_length_by_bond(identifier_ids, bond_num):
    """
    Get the length of input tokens with "bond_num" bonds in it.
    """
    bond_mask = ~identifier_ids
    bond_count = torch.cumsum(bond_mask, dim=-1)
    bond_mask_below = (bond_count <= bond_num)
    return torch.sum(bond_mask_below, dim=-1)


def results_are_same(result1, result2):
    if result1 is None or result2 is None:
        if result1 is None and result2 is None:
            return True
        else:
            return False
    else:
        if (
                torch.equal(result1["input_ids"], result2["input_ids"]) and
                torch.equal(result1["graph_position_ids_1"], result2["graph_position_ids_1"]) and
                torch.equal(result1["graph_position_ids_2"], result2["graph_position_ids_2"]) and
                torch.equal(result1["identifier_ids"], result2["identifier_ids"])
        ):
            return True
        else:
            return False


def main(args):
    if args.use_cache:
        raise NotImplementedError('"use_cache" is bugged, do not use it!')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    """load data & model"""
    with open(args.smiles_file, "r", encoding="utf-8") as f:
        smiles_list = f.readlines()
    smiles_list = [smiles.removesuffix("\n") for smiles in smiles_list]
    print(f"Total SMILES loaded: {len(smiles_list)}")

    property_info = load_json(args.property_info_file)  # 🔍
    tokenizer = GraphsGPTTokenizer.from_pretrained(args.model_name_or_path)
    model = GraphsGPTForConditionalGeneration.from_pretrained(args.model_name_or_path)  # 🔍
    model.to(device)

    """🔍 get global properties"""
    if args.value_qed is not None:  # qed
        normalized_qed = normalize_value(args.value_qed, mean=property_info["qed"]["mean"], std=property_info["qed"]["std"])
        values_qed = torch.full((args.batch_size, 1), fill_value=normalized_qed, device=device)
    else:
        values_qed = None

    if args.value_sa is not None:  # sa
        normalized_sa = normalize_value(args.value_sa, mean=property_info["sa"]["mean"], std=property_info["sa"]["std"])
        values_sa = torch.full((args.batch_size, 1), fill_value=normalized_sa, device=device)
    else:
        values_sa = None

    if args.value_logp is not None:  # logp
        normalized_logp = normalize_value(args.value_logp, mean=property_info["logp"]["mean"], std=property_info["logp"]["std"])
        values_logp = torch.full((args.batch_size, 1), fill_value=normalized_logp, device=device)
    else:
        values_logp = None

    """🔍 get global scaffold inputs"""
    if args.scaffold_smiles is not None and args.scaffold_smiles != "":
        with torch.no_grad():
            scaffold_inputs = tokenizer.encode(args.scaffold_smiles, return_tensors="pt")
            move_tensors_to_device(scaffold_inputs, device)

            # expand to shape(batch_size, scaffold_seq_len)
            scaffold_seq_len = scaffold_inputs["input_ids"].shape[0]
            scaffold_inputs["input_ids"] = scaffold_inputs["input_ids"].expand(args.batch_size, scaffold_seq_len)
            scaffold_inputs["graph_position_ids_1"] = scaffold_inputs["graph_position_ids_1"].expand(args.batch_size, scaffold_seq_len)
            scaffold_inputs["graph_position_ids_2"] = scaffold_inputs["graph_position_ids_2"].expand(args.batch_size, scaffold_seq_len)
            scaffold_inputs["identifier_ids"] = scaffold_inputs["identifier_ids"].expand(args.batch_size, scaffold_seq_len)
            scaffold_inputs["attention_mask"] = None
    else:
        scaffold_inputs = None

    """get initializing inputs"""
    if args.initialize_molecule_smiles is not None and args.initialize_bond_num > 0:  # initialization tokens
        initialize_molecule_input = tokenizer.encode(args.initialize_molecule_smiles, return_tensors="pt")
        move_tensors_to_device(initialize_molecule_input, device)

        max_initialize_bond_num = torch.sum(~initialize_molecule_input["identifier_ids"]).item()
        if args.initialize_bond_num >= max_initialize_bond_num:
            raise ValueError(f"Initializing bond num {max_initialize_bond_num} exceeds the maximum molecular bond length {max_initialize_bond_num}!")
        else:
            initialize_input_length = get_input_length_by_bond(initialize_molecule_input["identifier_ids"], args.initialize_bond_num)

            initialize_input_ids = initialize_molecule_input["input_ids"][:initialize_input_length]  # (initialize_input_length)
            initialize_input_ids = initialize_input_ids.reshape(1, 1, initialize_input_length)
            initialize_input_ids = initialize_input_ids.expand(1, args.batch_size, initialize_input_length)
            initialize_input_ids_list = torch.split(initialize_input_ids, 1, dim=1)  # list of tensors with shape (1, initialize_input_length)

            initialize_graph_position_ids_1 = initialize_molecule_input["graph_position_ids_1"][:initialize_input_length]  # (initialize_input_length)
            initialize_graph_position_ids_1 = initialize_graph_position_ids_1.reshape(1, 1, initialize_input_length)
            initialize_graph_position_ids_1 = initialize_graph_position_ids_1.expand(1, args.batch_size, initialize_input_length)
            initialize_graph_position_ids_1_list = torch.split(initialize_graph_position_ids_1, 1, dim=1)  # list of tensors with shape (1, initialize_input_length)

            initialize_graph_position_ids_2 = initialize_molecule_input["graph_position_ids_2"][:initialize_input_length]  # (initialize_input_length)
            initialize_graph_position_ids_2 = initialize_graph_position_ids_2.reshape(1, 1, initialize_input_length)
            initialize_graph_position_ids_2 = initialize_graph_position_ids_2.expand(1, args.batch_size, initialize_input_length)
            initialize_graph_position_ids_2_list = torch.split(initialize_graph_position_ids_2, 1, dim=1)  # list of tensors with shape (1, initialize_input_length)

            initialize_identifier_ids = initialize_molecule_input["identifier_ids"][:initialize_input_length]  # (initialize_input_length)
            initialize_identifier_ids = initialize_identifier_ids.reshape(1, 1, initialize_input_length)
            initialize_identifier_ids = initialize_identifier_ids.expand(1, args.batch_size, initialize_input_length)
            initialize_identifier_ids_list = torch.split(initialize_identifier_ids, 1, dim=1)  # list of tensors with shape (1, initialize_input_length)
    else:
        initialize_input_ids_list = None
        initialize_graph_position_ids_1_list = None
        initialize_graph_position_ids_2_list = None
        initialize_identifier_ids_list = None

    """start generation"""
    with torch.no_grad():
        for batch_idx, batched_smiles in enumerate(split_list_with_yield(smiles_list, args.batch_size)):
            """prepare inputs"""
            inputs = tokenizer.batch_encode(batched_smiles, return_tensors="pt")
            move_tensors_to_device(inputs, device)

            # 🔍 prepare this batch's properties
            mol_cache = {}
            this_batch_values_qed = values_qed
            this_batch_values_sa = values_sa
            this_batch_values_logp = values_logp

            if this_batch_values_qed is None:  # use the values of this batch if not specified globally
                this_batch_values_qed = []
                for smiles in batched_smiles:
                    if smiles not in mol_cache:
                        mol = Chem.MolFromSmiles(smiles)
                        mol_cache[smiles] = mol
                    this_batch_values_qed.append(normalize_value(get_qed(mol_cache[smiles]), mean=property_info["qed"]["mean"], std=property_info["qed"]["std"]))
                this_batch_values_qed = torch.tensor(this_batch_values_qed, device=device).reshape(args.batch_size, 1)

            if this_batch_values_sa is None:  # use the values of this batch if not specified globally
                this_batch_values_sa = []
                for smiles in batched_smiles:
                    if smiles not in mol_cache:
                        mol = Chem.MolFromSmiles(smiles)
                        mol_cache[smiles] = mol
                    this_batch_values_sa.append(normalize_value(get_sa(mol_cache[smiles]), mean=property_info["sa"]["mean"], std=property_info["sa"]["std"]))
                this_batch_values_sa = torch.tensor(this_batch_values_sa, device=device).reshape(args.batch_size, 1)

            if this_batch_values_logp is None:  # use the values of this batch if not specified globally
                this_batch_values_logp = []
                for smiles in batched_smiles:
                    if smiles not in mol_cache:
                        mol = Chem.MolFromSmiles(smiles)
                        mol_cache[smiles] = mol
                    this_batch_values_logp.append(normalize_value(get_logp(mol_cache[smiles]), mean=property_info["logp"]["mean"], std=property_info["logp"]["std"]))
                this_batch_values_logp = torch.tensor(this_batch_values_logp, device=device).reshape(args.batch_size, 1)

            # 🔍 prepare this batch's scaffolds
            this_batch_scaffold_inputs = scaffold_inputs

            if this_batch_scaffold_inputs is None:  # use the values of this batch if not specified globally
                scaffold_smiles = []
                for smiles in batched_smiles:
                    if smiles not in mol_cache:
                        mol = Chem.MolFromSmiles(smiles)
                        mol_cache[smiles] = mol
                    scaffold = get_molecule_standard_scaffold(mol_cache[smiles], normalizer=tokenizer.normalizer)
                    scaffold_smiles.append(tokenizer._convert_molecule_to_standard_smiles(scaffold))
                this_batch_scaffold_inputs = tokenizer.encode(scaffold_smiles, return_tensors="pt")
                move_tensors_to_device(this_batch_scaffold_inputs, device)

            # 🔍 add condition information
            inputs["scaffold_input_ids"] = this_batch_scaffold_inputs["input_ids"]
            inputs["scaffold_graph_position_ids_1"] = this_batch_scaffold_inputs["graph_position_ids_1"]
            inputs["scaffold_graph_position_ids_2"] = this_batch_scaffold_inputs["graph_position_ids_2"]
            inputs["scaffold_identifier_ids"] = this_batch_scaffold_inputs["identifier_ids"]
            inputs["scaffold_attention_mask"] = this_batch_scaffold_inputs["attention_mask"]
            inputs["values_qed"] = this_batch_values_qed
            inputs["values_sa"] = this_batch_values_sa
            inputs["values_logp"] = this_batch_values_logp

            for key in inputs.keys():
                print(key, inputs[key])

            """get fingerprint tokens"""
            fingerprint_tokens = model.encode_to_fingerprints(**inputs)  # (batch_size, num_fingerprints, hidden_dim)

            """generate molecules"""
            print(f"Generating for batch {batch_idx}...")
            generated_results: list = model.generate_from_fingerprints(
                fingerprint_tokens=fingerprint_tokens,
                bond_dict=tokenizer.bond_dict,
                input_ids_list=initialize_input_ids_list,
                graph_position_ids_1_list=initialize_graph_position_ids_1_list,
                graph_position_ids_2_list=initialize_graph_position_ids_2_list,
                identifier_ids_list=initialize_identifier_ids_list,
                strict_generation=args.strict_generation,
                do_sample=args.do_sample,
                topk=args.topk,
                temperature=args.temperature,
                max_atoms=None,
                similarity_threshold=args.similarity_threshold,
                check_first_node=args.check_first_node,
                check_atom_valence=args.check_atom_valence,
                fix_aromatic_bond=args.fix_aromatic_bond,
                use_cache=args.use_cache,
                save_failed=args.save_failed,
                show_progress=True,
                verbose=True  # You can disable the failure information by setting this to "False"
            )
            generated_results = [move_tensors_to_device(result, "cpu") for result in generated_results]

            """get ground truths"""
            groundtruth_results = []
            for i in range(args.batch_size):
                this_padding_mask = inputs["attention_mask"][i]  # (seq_len)
                groundtruth_results.append(
                    {
                        "input_ids": inputs["input_ids"][i][this_padding_mask].clone().cpu(),  # (this_seq_len)
                        "graph_position_ids_1": inputs["graph_position_ids_1"][i][this_padding_mask].clone().cpu(),  # (this_seq_len)
                        "graph_position_ids_2": inputs["graph_position_ids_2"][i][this_padding_mask].clone().cpu(),  # (this_seq_len)
                        "identifier_ids": inputs["identifier_ids"][i][this_padding_mask].clone().cpu(),  # (this_seq_len)
                    }
                )

            """save generated results"""
            if args.save_results:
                # paths
                save_dir_groundtruth = os.path.join(args.save_dir, "groundtruth_results")
                save_dir_generated = os.path.join(args.save_dir, "generated_results")

                create_dir(save_dir_groundtruth)
                create_dir(save_dir_generated)

                # save results
                save_file_groundtruth = os.path.join(save_dir_groundtruth, f"{batch_idx}.pkl")
                save_file_generated = os.path.join(save_dir_generated, f"{batch_idx}.pkl")

                with open(save_file_groundtruth, "wb") as f:
                    pickle.dump(groundtruth_results, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Groundtruth results of batch {batch_idx} saved to '{save_file_groundtruth}'!")

                with open(save_file_generated, "wb") as f:
                    pickle.dump(generated_results, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Generated results of batch {batch_idx} saved to '{save_file_generated}'!")

            """exit check"""
            if batch_idx >= args.num_batches - 1:
                break

    """save generation configuration"""
    # 🔍 property information
    save_file_properties = os.path.join(args.save_dir, "manual_properties.txt")
    with open(save_file_properties, "w") as f:
        if args.value_qed is not None:
            f.write(f"QED: {format(args.value_qed, '.2f')} (normalized value = {normalized_qed})\n")
        else:
            f.write(f"QED: None\n")

        if args.value_sa is not None:
            f.write(f"SA: {format(args.value_sa, '.2f')} (normalized value = {normalized_sa})\n")
        else:
            f.write(f"SA: None\n")

        if args.value_logp is not None:
            f.write(f"logP: {format(args.value_logp, '.2f')} (normalized value = {normalized_logp})\n")
        else:
            f.write(f"logP: None\n")
    print(f"Manual properties saved to '{save_file_properties}'!")

    # 🔍 scaffold information
    if args.scaffold_smiles is not None and args.scaffold_smiles != "":
        move_tensors_to_device(scaffold_inputs, "cpu")
        save_file_scaffold = os.path.join(args.save_dir, "manual_scaffold.pkl")
        with open(save_file_scaffold, "wb") as f:
            pickle.dump(scaffold_inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Reference molecule for fingerprint tokens sampling saved to '{save_file_scaffold}'!")

    # initializing information
    if args.initialize_molecule_smiles is not None and args.initialize_bond_num > 0:
        initialize_input = {
            "input_ids": initialize_input_ids_list[0].squeeze(0).clone().cpu(),  # (initialize_input_length)
            "graph_position_ids_1": initialize_graph_position_ids_1_list[0].squeeze(0).clone().cpu(),  # (initialize_input_length)
            "graph_position_ids_2": initialize_graph_position_ids_2_list[0].squeeze(0).clone().cpu(),  # (initialize_input_length)
            "identifier_ids": initialize_identifier_ids_list[0].squeeze(0).clone().cpu(),  # (initialize_input_length)
        }
        save_file_initialize = os.path.join(args.save_dir, "initialize_molecule.pkl")
        with open(save_file_initialize, "wb") as f:
            pickle.dump(initialize_input, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Initializing molecule saved to '{save_file_initialize}'!")

    print("Reached max batch number limit, stopping generation...")
    print("Done!")


if __name__ == '__main__':
    parser = ArgumentParser()

    # Basic Information
    parser.add_argument("--model_name_or_path", default="DaizeDong/GraphsGPT-1W", type=str)
    parser.add_argument('--save_dir', default="./results/conditional/generation", type=str)
    parser.add_argument('--smiles_file', default="./data/examples/zinc_example.txt", type=str)
    parser.add_argument('--num_batches', default=10, type=int, help="Number of batches to generate.")
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # Generation Initialize Tokens
    parser.add_argument('--initialize_molecule_smiles', default=None, type=str, help="SMILES of the molecule for initializing the generation sequence.")
    parser.add_argument('--initialize_bond_num', default=0, type=int)

    # 🔍 Generation Property Information
    parser.add_argument('--property_info_file', default="./property_info.json", type=str, help="File that records the mean & variance of each property, used to normalize the input")  # 🔍
    parser.add_argument('--value_qed', default=None, type=float)
    parser.add_argument('--value_sa', default=None, type=float)
    parser.add_argument('--value_logp', default=None, type=float)
    parser.add_argument('--scaffold_smiles', default=None, type=str)

    # Generation Configurations
    parser.add_argument('--strict_generation', default="True", type=str)
    parser.add_argument('--do_sample', default="False", type=str)
    parser.add_argument('--topk', default=None, type=int)
    parser.add_argument('--temperature', default=1.0, type=float)
    parser.add_argument('--similarity_threshold', default=0.5, type=float)
    parser.add_argument('--check_first_node', default="True", type=str)
    parser.add_argument('--check_atom_valence', default="False", type=str)
    parser.add_argument('--fix_aromatic_bond', default="False", type=str)
    parser.add_argument('--use_cache', default="False", type=str)
    parser.add_argument('--save_results', default="True", type=str)
    parser.add_argument('--save_failed', default="False", type=str)

    args = parser.parse_args()
    args.strict_generation = str2bool(args.strict_generation)
    args.do_sample = str2bool(args.do_sample)
    args.check_first_node = str2bool(args.check_first_node)
    args.check_atom_valence = str2bool(args.check_atom_valence)
    args.use_cache = str2bool(args.use_cache)
    args.save_results = str2bool(args.save_results)
    args.save_failed = str2bool(args.save_failed)

    print("CUDA available: ", torch.cuda.is_available())

    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)
