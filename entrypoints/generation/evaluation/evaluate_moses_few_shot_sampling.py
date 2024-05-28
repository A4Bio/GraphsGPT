import os.path

import argparse
import random
import torch
from tqdm import tqdm

import moses
from data.collate_fn import tensor_dict_stack_padding_collater
from data.tokenizer import GraphsGPTTokenizer
from models.graphsgpt.modeling_graphsgpt import GraphsGPTForCausalLM
from utils.io import create_dir, save_json, delete_file_or_dir
from utils.operations.operation_list import split_list
from utils.operations.operation_tensor import move_tensors_to_device


def main(args):
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """load data & model"""
    train_smiles = moses.get_dataset("train")
    # test_smiles = moses.get_dataset("test")
    # test_scaffolds = moses.get_dataset("test_scaffolds")

    pad_collator = tensor_dict_stack_padding_collater(0, tensor_keys_to_create_mask=["input_ids"])

    tokenizer = GraphsGPTTokenizer.from_pretrained(args.model_name_or_path)
    model = GraphsGPTForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    """generate according to reference fingerprints"""
    all_smiles = []

    # initialize the variable for batch-optimization of few-shot generation
    max_shots_in_a_batch = args.batch_size_valid // args.num_samples_each_shot
    next_shot_id = 0
    finished_shots = set()  # shots that have done the generation
    candidate_shots = {}  # shots for the next generation forward

    with torch.inference_mode():
        while len(finished_shots) < args.num_shots:
            """prepare initial fingerprint tokens"""
            extra_inputs_num = max_shots_in_a_batch - len(candidate_shots)

            if extra_inputs_num > 0:  # need some new inputs to fill up the batch
                # construct inputs
                inputs = []
                for i in range(next_shot_id, next_shot_id + extra_inputs_num):
                    this_shot_inputs = tokenizer.encode(train_smiles[i], return_tensors="pt")
                    if this_shot_inputs is None:  # the encoding may fail
                        extra_inputs_num -= 1
                        continue
                    move_tensors_to_device(this_shot_inputs, device)
                    inputs.append(this_shot_inputs)
                inputs, mask = pad_collator(inputs)
                inputs["attention_mask"] = mask["input_ids"]

                # repeat fingerprint tokens
                fingerprint_tokens = model.encode_to_fingerprints(**inputs)
                fingerprint_tokens = fingerprint_tokens.repeat_interleave(args.num_samples_each_shot, dim=0)  # (extra_inputs_num * num_samples_each_shot, num_fingerprints, hidden_size)
                each_shot_fingerprint_tokens = fingerprint_tokens.split(args.num_samples_each_shot, dim=0)  # each is (num_samples_each_shot, num_fingerprints, hidden_size)

                # add to shots
                for shot_id, this_shot_fingerprint_tokens in enumerate(each_shot_fingerprint_tokens, start=next_shot_id):
                    candidate_shots[shot_id] = {
                        "sample_times": 0,
                        "fingerprint": this_shot_fingerprint_tokens,
                        "generation_results": [],
                    }
                print(f"Total {len(candidate_shots)} candidate shots in this iteration!")

                # update next_shot_id
                next_shot_id += extra_inputs_num
                print(f"Encoded {extra_inputs_num} new fingerprints!")

                """aggregate fingerprint tokens for candidate shots"""
                shot_ids_in_batch = []
                aggregated_fingerprint_tokens = []
                for shot_id, candidate_shot in candidate_shots.items():
                    shot_ids_in_batch.append(shot_id)
                    aggregated_fingerprint_tokens.append(candidate_shot["fingerprint"])
                aggregated_fingerprint_tokens = torch.cat(aggregated_fingerprint_tokens, dim=0)  # (max_shots_in_a_batch * num_samples_each_shot, num_fingerprints, hidden_size)

            """random sampling & generate"""
            # For each shot, we try to randomly sample fingerprints for at most (max_sample_times * num_samples_each_shot) times.
            # If the sampling trys exceed the limit, we stop sampling this shot.
            generate_fingerprint_tokens = torch.normal(mean=aggregated_fingerprint_tokens, std=args.sample_std)
            generated_results: list = model.generate_from_fingerprints(
                fingerprint_tokens=generate_fingerprint_tokens,
                bond_dict=tokenizer.bond_dict,
                input_ids_list=None,
                graph_position_ids_1_list=None,
                graph_position_ids_2_list=None,
                identifier_ids_list=None,
                strict_generation=True,
                do_sample=False,
                topk=1,
                temperature=1.0,
                max_atoms=None,
                similarity_threshold=0.5,
                check_first_node=True,
                check_atom_valence=True,
                fix_aromatic_bond=True,
                use_cache=False,
                save_failed=False,
                show_progress=True,
                verbose=False,
            )
            each_shot_generate_results = split_list(generated_results, args.num_samples_each_shot)

            """add results to corresponding shots"""
            for shot_id, this_shot_generate_results in zip(shot_ids_in_batch, each_shot_generate_results):
                this_shot_generate_results = [move_tensors_to_device(result, "cpu") for result in this_shot_generate_results if result is not None]  # remove None results
                candidate_shots[shot_id]["generation_results"].extend(this_shot_generate_results)
                candidate_shots[shot_id]["sample_times"] += 1
                print(f"Added {len(this_shot_generate_results)} generated results for shot {shot_id}. Now: {len(candidate_shots[shot_id]['generation_results'])}")

            """gather results & remove finished shots"""
            this_iter_finished_shot_ids = []
            this_iter_finished_results = []
            for shot_id, candidate_shot in tqdm(candidate_shots.items(), desc="Aggregating results"):
                if candidate_shot["sample_times"] >= args.max_sample_times or len(candidate_shots[shot_id]["generation_results"]) >= args.num_samples_each_shot:  # exceed max trys / results are enough
                    this_iter_finished_shot_ids.append(shot_id)
                    this_iter_finished_results.extend(candidate_shots[shot_id]["generation_results"][:args.num_samples_each_shot])  # add at most "num_samples_each_shot" results
                if len(finished_shots) + len(this_iter_finished_shot_ids) >= args.num_shots:
                    break

            for shot_id in this_iter_finished_shot_ids:
                print(f"Finished shot {shot_id}. Final samples: {len(candidate_shots[shot_id]['generation_results'])}")
                candidate_shots.pop(shot_id)  # remove finished shots
                finished_shots.add(shot_id)
            print(f"Now total finished shots: {len(finished_shots)}")

            print("Decoding to SMILES...")
            if len(this_iter_finished_results) > 0:
                this_iter_finished_results_batched, mask = pad_collator(this_iter_finished_results)
                this_iter_finished_results_batched["attention_mask"] = mask["input_ids"]
                decoded_mol_list, decoded_smiles_list = tokenizer.decode(this_iter_finished_results_batched, kekulize=True, nprocs=None)
                # You can use the following line to accelerate the decoding. However, it may occasionally raise errors.
                # decoded_mol_list, decoded_smiles_list = tokenizer.decode(this_iter_finished_results_batched, kekulize=True, nprocs=args.num_processes)
                all_smiles.extend(decoded_smiles_list)
            print(f"Now total generated SMILES: {len(all_smiles)}")

    """get metrics"""
    print("Getting metrics...")
    metrics = moses.get_all_metrics(
        all_smiles,
        n_jobs=args.num_processes,
        device=device,
        batch_size=args.batch_size_valid,
    )
    print(metrics)

    delete_file_or_dir(args.save_path)
    create_dir(args.save_path)
    save_metric_file = os.path.join(args.save_path, "metrics.json")
    save_json(metrics, save_metric_file)
    print(f"Metrics saved to {save_metric_file}")

    """save results"""
    save_all_smiles_file = os.path.join(args.save_path, f"all_smiles.txt")
    with open(save_all_smiles_file, "w") as f:
        for smiles in all_smiles:
            f.write(smiles + "\n")
    print(f"Generated SMILES saved to {save_all_smiles_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="DaizeDong/GraphsGPT-1W", type=str, help="Path to the GraphsGPT hugging face model.")
    parser.add_argument("--save_path", default="./results/unconditional/moses", type=str, help="Path to save the evaluation results.")

    parser.add_argument('--batch_size_valid', default=8192, type=int, help="Number of samples per batch.")
    parser.add_argument("--sample_std", default=1.0, type=float, help="The standard deviation for sampling.")
    parser.add_argument('--max_sample_times', default=10, type=int, help="The maximum number of attempts to sample for each shot. Shots with insufficient successful generated results exceeding this number of attempts will be discarded.")
    parser.add_argument("--num_shots", default=100000, type=int, help="The number of shots for reference.")
    parser.add_argument("--num_samples_each_shot", default=1, type=int, help="The number of generated samples for each shot.")

    parser.add_argument("--num_processes", default=32, type=int, help="Number of parallel processes for decoding & metric calculation.")
    args = parser.parse_args()

    print(args)
    main(args)
