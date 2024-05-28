import os
import pickle
import torch
from argparse import ArgumentParser

from data.tokenizer import GraphsGPTTokenizer
from models.graphsgpt.modeling_graphsgpt import GraphsGPTForCausalLM
from utils.io import create_dir, get_avg_acc_from_file, delete_file_or_dir
from utils.operations.operation_list import split_list_with_yield
from utils.operations.operation_string import str2bool
from utils.operations.operation_tensor import move_tensors_to_device


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

    tokenizer = GraphsGPTTokenizer.from_pretrained(args.model_name_or_path)
    model = GraphsGPTForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

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
    save_file_match_num = os.path.join(args.save_dir, "match_num.txt")
    delete_file_or_dir(save_file_match_num)  # remove existing summary results

    with torch.no_grad():
        for batch_idx, batched_smiles in enumerate(split_list_with_yield(smiles_list, args.batch_size)):
            """prepare inputs"""
            inputs = tokenizer.batch_encode(batched_smiles, return_tensors="pt")
            move_tensors_to_device(inputs, device)

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

            """save results"""
            if args.save_results:
                """basic results"""
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

            """save accuracy"""
            # (this is the absolute accuracy that generated results completely match the ground truth results)
            correct_num = 0
            for i in range(args.batch_size):
                if results_are_same(groundtruth_results[i], generated_results[i]):
                    correct_num += 1
            acc = correct_num / args.batch_size

            with open(save_file_match_num, "a") as f:
                f.write(f"Accuracy of batch {batch_idx}: {format(acc * 100, '.2f')}% ({correct_num}/{args.batch_size})\n")
            print(f"Accuracy of batch {batch_idx}: {format(acc * 100, '.2f')}% ({correct_num}/{args.batch_size})\n")

            """exit check"""
            if batch_idx >= args.num_batches - 1:
                break

    """save generation configuration"""
    # summary accuracy
    avg_acc = get_avg_acc_from_file(save_file_match_num)

    with open(save_file_match_num, "a") as f:
        f.write(f"Average accuracy: {format(avg_acc, '.2f')}%\n")
    print(f"Average accuracy: {format(avg_acc, '.2f')}%\n")

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
    parser.add_argument('--save_dir', default="./results/unconditional/generation", type=str)
    parser.add_argument('--smiles_file', default="./data/examples/zinc_example.txt", type=str)
    parser.add_argument('--num_batches', default=10, type=int, help="Number of batches to generate.")
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--seed', default=0, type=int)

    # Generation Initialize Tokens
    parser.add_argument('--initialize_molecule_smiles', default=None, type=str, help="SMILES of the molecule for initializing the generation sequence.")
    parser.add_argument('--initialize_bond_num', default=0, type=int)

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
