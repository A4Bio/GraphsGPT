import csv
import gzip
import json
import lzma
import numpy as np
import os
import pickle
import shutil
from rdkit import Chem
from rdkit.Chem import Draw
from typing import Union, List, Dict

from utils.operations.operation_string import extract_numbers


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def delete_file_or_dir(dir):
    if os.path.isfile(dir):
        os.remove(dir)
    elif os.path.exists(dir):
        shutil.rmtree(dir)
    else:
        pass


def save_compressed_file_7z(data, file_path):  # 7z
    create_dir(os.path.dirname(file_path))
    with lzma.open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_compressed_file_7z(file_path):  # 7z
    with lzma.open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def save_compressed_file_gz(data, file_path, compresslevel=6):  # gz
    create_dir(os.path.dirname(file_path))
    with gzip.open(file_path, "wb", compresslevel=compresslevel) as file:
        pickle.dump(data, file)


def load_compressed_file_gz(file_path):  # gz
    with gzip.open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def read_csv(file_path, has_header=True) -> Union[List[List], List[Dict]]:
    """
    Read a CSV file and return its content.

    Args:
    - file_path (str): Path to the CSV file.
    - has_header (bool): Whether the CSV file has a header. Default is True.

    Returns:
    - list of list or dict: Content of the CSV file.
      If has_header is True, return a list of dictionaries;
      if has_header is False, return a list of lists.
    """
    data = []
    with open(file_path, newline='', encoding='utf-8') as f:
        if has_header:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                data.append(dict(row))
        else:
            csvreader = csv.reader(f)
            for row in csvreader:
                data.append(row)
    return data


def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path, indent=4, **kwargs):
    create_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


def load_jsonl(file_path) -> list:
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}")
                continue
    return data


def save_jsonl(data, file_path, **kwargs):
    create_dir(os.path.dirname(file_path))
    with open(file_path, "w", encoding="utf8") as f:
        for ins in data:
            f.write(f"{json.dumps(ins, ensure_ascii=False, **kwargs)}\n")


def compress_png_image(image_path, print_info=False):
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if print_info:
        print(f'Done for "{image_path}".')


"""for this project"""


def find_best_model(model_dir):
    best_model_file = None
    best_opt_steps = 0
    best_val_loss = float("inf")

    for root, dirs, files in os.walk(model_dir):
        for file in files:
            file_postfix = file.split(".")[-1]
            file_name = file.replace("." + file_postfix, "")

            if file_postfix == "ckpt" and "best" in file_name:
                # Example: "best-opt_steps=091999-val_loss=0.1109.ckpt"
                this_opt_steps, this_val_loss = extract_numbers(file_name)

                if this_val_loss < best_val_loss:
                    # model with the minimal val_loss
                    best_model_file = os.path.join(root, file)
                    best_opt_steps = this_opt_steps
                    best_val_loss = this_val_loss

                elif this_val_loss == best_val_loss and this_opt_steps > best_opt_steps:
                    # model with the largest opt_steps
                    best_model_file = os.path.join(root, file)
                    best_opt_steps = this_opt_steps
                    best_val_loss = this_val_loss

    return best_model_file


def find_last_model(model_dir):
    last_model_file = None

    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if file == "last.ckpt":
                last_model_file = os.path.join(root, file)
                break

    return last_model_file


def get_avg_acc_from_file(save_file_match_num):
    acc_list = []
    with open(save_file_match_num, "r") as f:
        lines = f.readlines()
        for line in lines:
            # example line: "Accuracy of sample 0: 100.00% (1024/1024)"
            # example line: "Consistency of sample 0: 100.00% (523776/523776)"
            numbers = extract_numbers(line)
            acc_list.append(numbers[1])
    return sum(acc_list) / len(acc_list)


def save_mol_png(mol, save_path, size=(512, 512)):
    img = Draw.MolToImage(mol, size=size)
    img.save(save_path)
    img.close()


def save_empty_png(save_path, size=(512, 512)):
    img = Draw.MolToImage(Chem.Mol(), size=size)
    img.save(save_path)
    img.close()


def summary_property_from_file(file_name):
    value_list = []
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.removesuffix("\n")
            if line != "None":
                number = float(line)
                value_list.append(number)
    if len(value_list) > 0:
        values = np.array(value_list)
        return values.mean(), values.std(), len(value_list)
    else:
        return -1, -1, -1


def summary_property_from_all_files(search_dir, file_name, value_type="float"):
    value_list = []
    for root, dirs, files in os.walk(search_dir):
        for file in files:
            if file == file_name:
                with open(os.path.join(root, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.removesuffix("\n")
                        if line != "None":
                            if value_type == "float":
                                number = float(line)
                            elif value_type == "bool":
                                number = (line == "True")
                            value_list.append(number)
    if len(value_list) > 0:
        values = np.array(value_list)
        return values.mean(), values.std(), len(value_list)
    else:
        return -1, -1, -1
