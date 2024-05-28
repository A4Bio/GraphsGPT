import os
import pickle
from argparse import ArgumentParser
from rdkit import RDLogger
from tqdm import tqdm

from data.tokenizer import GraphsGPTTokenizer
from utils.io import delete_file_or_dir, create_dir, save_mol_png, save_empty_png, summary_property_from_file, summary_property_from_all_files
from utils.molecule import get_molecule_standard_scaffold
from utils.operations.operation_string import str2bool
from utils.property_scores.scoring_func import get_qed, get_sa, get_logp, get_is_valid

RDLogger.DisableLog('rdApp.*')


def main(args):
    tokenizer = GraphsGPTTokenizer.from_pretrained(args.model_name_or_path)

    file_list = os.listdir(args.generation_results_dir)
    file_list = sorted(file_list)
    file_list = file_list[args.file_begin_index:args.file_end_index]

    all_smiles = []

    delete_file_or_dir(args.save_dir)

    """visualize"""
    for file_name in tqdm(file_list):
        success_cnt = 0
        invalid_cnt = 0
        fail_cnt = 0

        valid_list = []
        qed_list = []
        sa_list = []
        logp_list = []
        smiles_list = []
        scaffold_smiles_list = []

        save_dir = os.path.join(args.save_dir, f"{file_name.split('.')[0]}")
        create_dir(save_dir)

        """read file"""
        file_path = os.path.join(args.generation_results_dir, file_name)
        with open(file_path, 'rb') as f:
            result_list = pickle.load(f)
        if not isinstance(result_list, (list, tuple)):
            result_list = [result_list]

        """save png"""
        for i, result in enumerate(result_list):
            print(i)
            save_img_file = os.path.join(save_dir, f"{i}.png")

            if result is not None:
                mol, smiles = tokenizer.decode(result, kekulize=True)  # kekulize the molecule for score calculation

                if mol is None:
                    valid_list.append(None)
                    qed_list.append(None)
                    sa_list.append(None)
                    logp_list.append(None)
                    smiles_list.append(None)
                    scaffold_smiles_list.append(None)
                    if args.save_images:
                        save_empty_png(save_img_file)
                    invalid_cnt += 1
                else:
                    valid = get_is_valid(mol)
                    qed = get_qed(mol)
                    sa = get_sa(mol)
                    logp = get_logp(mol)

                    scaffold = get_molecule_standard_scaffold(mol, normalizer=tokenizer.normalizer)
                    scaffold_smiles = tokenizer._convert_molecule_to_standard_smiles(scaffold)

                    valid_list.append(valid)
                    qed_list.append(qed)
                    sa_list.append(sa)
                    logp_list.append(logp)
                    smiles_list.append(smiles)
                    scaffold_smiles_list.append(scaffold_smiles)

                    if args.save_images:
                        save_mol_png(mol, save_img_file)
                    success_cnt += 1
            else:
                valid_list.append(None)
                qed_list.append(None)
                sa_list.append(None)
                logp_list.append(None)
                smiles_list.append(None)
                scaffold_smiles_list.append(None)
                if args.save_images:
                    save_empty_png(save_img_file)
                fail_cnt += 1

        all_smiles.extend(smiles_list)

        """save statistics"""
        with open(os.path.join(save_dir, "count.txt"), 'a') as f:
            f.write(f"Success count: {success_cnt}\n")
            f.write(f"Invalid count: {invalid_cnt}\n")
            f.write(f"Fail count: {fail_cnt}\n")

        with open(os.path.join(save_dir, "valid.txt"), 'a') as f:
            for valid in valid_list:
                f.write(f"{valid}\n")

        with open(os.path.join(save_dir, "qed.txt"), 'a') as f:
            for qed in qed_list:
                f.write(f"{qed}\n")

        with open(os.path.join(save_dir, "sa.txt"), 'a') as f:
            for sa in sa_list:
                f.write(f"{sa}\n")

        with open(os.path.join(save_dir, "logp.txt"), 'a') as f:
            for logp in logp_list:
                f.write(f"{logp}\n")

        with open(os.path.join(save_dir, "smiles.txt"), 'a') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")

        with open(os.path.join(save_dir, "scaffold_smiles.txt"), 'a') as f:
            for scaffold_smiles in scaffold_smiles_list:
                f.write(f"{scaffold_smiles}\n")

        mean_qed, std_qed, num_qed = summary_property_from_file(os.path.join(save_dir, "qed.txt"))
        mean_sa, std_sa, num_sa = summary_property_from_file(os.path.join(save_dir, "sa.txt"))
        mean_logp, std_logp, num_logp = summary_property_from_file(os.path.join(save_dir, "logp.txt"))

        valid_num = sum([1 for valid in valid_list if valid])
        total_num = len([smiles for smiles in smiles_list if smiles is not None])

        with open(os.path.join(save_dir, "summary.txt"), 'w') as f:
            f.write(f"Summary QED: mean={format(mean_qed, '.3f')}, std={format(std_qed, '.3f')}, total_cnt={num_qed}\n")
            f.write(f"Summary SA: mean={format(mean_sa, '.3f')}, std={format(std_sa, '.3f')}, total_cnt={num_sa}\n")
            f.write(f"Summary logP: mean={format(mean_logp, '.3f')}, std={format(std_logp, '.3f')}, total_cnt={num_logp}\n")
            f.write(f"Summary validity: {format(valid_num / total_num * 100, '.2f')}% ({valid_num}/{total_num})\n")

    """summarize the results"""
    mean_qed, std_qed, num_qed = summary_property_from_all_files(args.save_dir, "qed.txt", value_type="float")
    mean_sa, std_sa, num_sa = summary_property_from_all_files(args.save_dir, "sa.txt", value_type="float")
    mean_logp, std_logp, num_logp = summary_property_from_all_files(args.save_dir, "logp.txt", value_type="float")
    mean_validity, std_validity, num_validity = summary_property_from_all_files(args.save_dir, "valid.txt", value_type="bool")

    all_total_num = len([smiles for smiles in all_smiles if smiles is not None])

    with open(os.path.join(args.save_dir, "final_summary.txt"), 'w') as f:
        f.write(f"Summary QED: mean={format(mean_qed, '.3f')}, std={format(std_qed, '.3f')}, total_cnt={num_qed}\n")
        f.write(f"Summary SA: mean={format(mean_sa, '.3f')}, std={format(std_sa, '.3f')}, total_cnt={num_sa}\n")
        f.write(f"Summary logP: mean={format(mean_logp, '.3f')}, std={format(std_logp, '.3f')}, total_cnt={num_logp}\n")
        f.write(f"Summary validity: {format(mean_validity * 100, '.2f')}% ({round(mean_validity * all_total_num)}/{all_total_num})\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", default="DaizeDong/GraphsGPT-1W-C", type=str)
    parser.add_argument('--generation_results_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--save_images', default="True", type=str)

    parser.add_argument('--file_begin_index', default=0, type=int)
    parser.add_argument('--file_end_index', default=2, type=int)
    args = parser.parse_args()
    args.save_images = str2bool(args.save_images)
    print(args)
    main(args)
