import os
import pickle
from argparse import ArgumentParser
from tqdm import tqdm

from data.tokenizer import GraphsGPTTokenizer
from utils.io import delete_file_or_dir, create_dir, save_mol_png, save_empty_png
from utils.operations.operation_string import str2bool


def main(args):
    tokenizer = GraphsGPTTokenizer.from_pretrained(args.model_name_or_path)

    file_list = os.listdir(args.generation_results_dir)
    file_list = sorted(file_list)
    file_list = file_list[args.file_begin_index:args.file_end_index]

    """visualize"""
    for file_name in tqdm(file_list):
        success_cnt = 0
        invalid_cnt = 0
        fail_cnt = 0
        smiles_list = []

        save_dir = os.path.join(args.save_dir, f"{file_name.split('.')[0]}")
        delete_file_or_dir(save_dir)
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
                    smiles_list.append(None)
                    if args.save_images:
                        save_empty_png(save_img_file)
                    invalid_cnt += 1
                else:
                    smiles_list.append(smiles)
                    if args.save_images:
                        save_mol_png(mol, save_img_file)
                    # print(f"Molecule '{smiles}' saved to '{save_img_file}'.")
                    success_cnt += 1
            else:
                smiles_list.append(None)
                if args.save_images:
                    save_empty_png(save_img_file)
                fail_cnt += 1

        """save statistics"""
        with open(os.path.join(save_dir, "count.txt"), 'a') as f:
            f.write(f"Success count: {success_cnt}\n")
            f.write(f"Invalid count: {invalid_cnt}\n")
            f.write(f"Fail count: {fail_cnt}\n")

        with open(os.path.join(save_dir, "smiles.txt"), 'a') as f:
            for smiles in smiles_list:
                f.write(f"{smiles}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", default="DaizeDong/GraphsGPT-1W", type=str)
    parser.add_argument('--generation_results_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--save_images', default="True", type=str)

    parser.add_argument('--file_begin_index', default=0, type=int)
    parser.add_argument('--file_end_index', default=2, type=int)
    args = parser.parse_args()
    args.save_images = str2bool(args.save_images)
    print(args)
    main(args)
