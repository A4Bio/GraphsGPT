import torch
import torch.nn.utils.rnn as rnn_utils
from typing import List


def identity_collator(examples):
    return examples


def tensor_stack_collator(examples):
    """
    examples: list of tensors.
    input: [tensor1, tensor2, ..., tensorN]
    output: stacked_tensor
    """
    return torch.stack(examples, dim=0)


class tensor_stack_padding_collater:
    """
    examples: list of tensors.
    input: [tensor1, tensor2, ..., tensorN]
    output: padded_tensor
    """

    def __init__(self, padding_id, padding_position="right", return_padding_mask=True):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

    def __call__(self, examples):
        dtype = examples[0].dtype
        if self.padding_position == "right":
            padded_examples = rnn_utils.pad_sequence(examples, batch_first=True, padding_value=self.padding_id)
        elif self.padding_position == "left":  # This will take about twice the time compared to right padding
            flipped_examples = [torch.flip(tensor, dims=[0]) for tensor in examples]
            padded_examples_flip = rnn_utils.pad_sequence(flipped_examples, batch_first=True, padding_value=self.padding_id)
            padded_examples = torch.flip(padded_examples_flip, dims=[1])
        else:
            raise NotImplementedError
        padded_examples = padded_examples.to(dtype)

        if self.return_padding_mask:
            padding_mask = (padded_examples != self.padding_id)
            return padded_examples, padding_mask
        else:
            return padded_examples


def tensor_lists_stack_collator(examples):
    """
    examples: list of tensor lists.
    input:
    [
        [tensor1, tensor1, ..., tensor1],
        [tensor2, tensor2, ..., tensor2],
        ...
        [tensorN, tensorN, ..., tensorN],
    ]
    output:
    [
        stacked_tensor1,
        stacked_tensor2,
        ...
        stacked_tensorN,
    ]
    """
    return [torch.stack([tensor_list[i] for tensor_list in examples], dim=0) for i in range(len(examples[0]))]


class tensor_lists_stack_padding_collater:
    def __init__(self, padding_id, padding_position="right", return_padding_mask=True, tensor_ids_to_create_mask: List = None):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

        # set indices of tensors in list to create "padding_mask"
        # if set to "None", then "padding_mask" will be created for all keys in dict
        self.tensor_ids_to_create_mask = tensor_ids_to_create_mask

    def __call__(self, examples):
        """
        examples: list of tensor lists.
        input:
        [
            [tensor1, tensor1, ..., tensor1],
            [tensor2, tensor2, ..., tensor2],
            ...
            [tensorN, tensorN, ..., tensorN],
        ]
        output:
        [
            padded_tensor1,
            padded_tensor2,
            ...
            padded_tensorN,
        ]
        """
        num_tensors = len(examples[0])
        padded_tensors = []
        padding_masks = []

        for i in range(num_tensors):
            dtype = examples[0][0].dtype
            if self.padding_position == "right":
                tensors = [tensor_list[i] for tensor_list in examples]
                padded_tensor = rnn_utils.pad_sequence(tensors, batch_first=True, padding_value=self.padding_id)
            elif self.padding_position == "left":  # This will take about twice the time compared to right padding
                flipped_tensors = [torch.flip(tensor_list[i], dims=[0]) for tensor_list in examples]
                flipped_padded_tensors = rnn_utils.pad_sequence(flipped_tensors, batch_first=True, padding_value=self.padding_id)
                padded_tensor = torch.flip(flipped_padded_tensors, dims=[1])
            else:
                raise NotImplementedError
            padded_tensor = padded_tensor.to(dtype)

            padded_tensors.append(padded_tensor)

            if self.return_padding_mask:
                if self.tensor_ids_to_create_mask is None or i in self.tensor_ids_to_create_mask:
                    padding_masks.append(padded_tensors[i] != self.padding_id)
                else:
                    padding_masks.append(None)

        if self.return_padding_mask:
            return padded_tensors, padding_masks
        else:
            return padded_tensors


def tensor_dicts_stack_collator(examples):
    """
    examples: list of tensor dicts.
    input:
    [
        {
            "key1": tensor1,
            "key2": tensor2,
            ...
            "keyN": tensorN,
        },
        {
            "key1": tensor1,
            "key2": tensor2,
            ...
            "keyN": tensorN,
        },
        ...
        {
            "key1": tensor1,
            "key2": tensor2,
            ...
            "keyN": tensorN,
        }
    ]
    output:
    {
        "key1": stacked_tensor1,
        "key2": stacked_tensor2,
        ...
        "keyN": stacked_tensorN,
    }
    """
    return {key: torch.stack([tensor_dict[key] for tensor_dict in examples], dim=0) for key in examples[0].keys()}


class tensor_dict_stack_padding_collater:
    def __init__(self, padding_id, padding_position="right", return_padding_mask=True, tensor_keys_to_create_mask: List = None):
        assert padding_position in ("left", "right")
        self.padding_id = padding_id
        self.padding_position = padding_position
        self.return_padding_mask = return_padding_mask

        # set keys of tensors in dict to create "padding_mask"
        # if set to "None", then "padding_mask" will be created for all keys in dict
        self.tensor_keys_to_create_mask = tensor_keys_to_create_mask

    def __call__(self, examples):
        """
        examples: list of tensor (or other types) dicts.
        input:
        [
            {
                "key0": int,
                "key1": tensor1,
                "key2": tensor2,
                ...
                "keyN": tensorN,
            },
            {
                "key0": int,
                "key1": tensor1,
                "key2": tensor2,
                ...
                "keyN": tensorN,
            },
            ...
            {
                "key0": int,
                "key1": tensor1,
                "key2": tensor2,
                ...
                "keyN": tensorN,
            }
        ]
        output:
        {
            "key0": [int, int, ..., int],
            "key1": padded_tensor1,
            "key2": padded_tensor2,
            ...
            "keyN": padded_tensorN,
        }
        """
        keys = examples[0].keys()
        padded_tensors = {}
        padding_masks = {}

        for key in keys:
            if isinstance(examples[0][key], torch.Tensor):
                if self.padding_position == "right":
                    tensors = [tensor_dict[key] for tensor_dict in examples]
                    padded_tensor = rnn_utils.pad_sequence(tensors, batch_first=True, padding_value=self.padding_id)
                elif self.padding_position == "left":  # This will take about twice the time compared to right padding
                    flipped_tensors = [torch.flip(tensor_dict[key], dims=[0]) for tensor_dict in examples]
                    flipped_padded_tensors = rnn_utils.pad_sequence(flipped_tensors, batch_first=True, padding_value=self.padding_id)
                    padded_tensor = torch.flip(flipped_padded_tensors, dims=[1])
                else:
                    raise NotImplementedError
            else:  # not tensor type, return as a list
                padded_tensor = [tensor_dict[key] for tensor_dict in examples]

            padded_tensors[key] = padded_tensor

            if self.return_padding_mask and isinstance(examples[0][key], torch.Tensor):
                if self.tensor_keys_to_create_mask is None or key in self.tensor_keys_to_create_mask:
                    padding_masks[key] = (padded_tensors[key] != self.padding_id)
                else:
                    padding_masks[key] = None

        if self.return_padding_mask:
            return padded_tensors, padding_masks
        else:
            return padded_tensors
