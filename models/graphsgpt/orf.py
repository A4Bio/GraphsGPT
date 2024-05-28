"""
Modified from https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def orthogonal_matrix_chunk(cols, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.transpose(0, 1)  # [cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix(nb_columns, nb_rows, random_shuffle=False, device=None, dtype=torch.float32):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        # q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(torch.zeros((nb_columns, remaining_rows), device=device))

    final_matrix = torch.cat(block_list, dim=1).type(dtype)
    final_matrix = F.normalize(final_matrix, p=2, dim=1)

    if random_shuffle:
        _, indices = torch.rand((final_matrix.shape[1],), device=device).sort(dim=0)
        indices = indices.unsqueeze(0).expand(final_matrix.shape)
        final_matrix = torch.gather(final_matrix, dim=1, index=indices)

    return final_matrix  # (nb_columns, nb_rows)


@torch.no_grad()
def orthogonal_matrix_chunk_batched(bsz, cols, device=None):
    unstructured_block = torch.randn((bsz, cols, cols), device=device)
    q, r = torch.linalg.qr(unstructured_block, mode='reduced')
    return q.transpose(1, 2)  # [bsz, cols, cols]


@torch.no_grad()
def gaussian_orthogonal_random_matrix_batched(nb_samples, nb_columns, nb_rows, random_shuffle=False, device=None, dtype=torch.float32):
    """create 2D Gaussian orthogonal matrix"""
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        # q = orthogonal_matrix_chunk_batched(nb_samples, nb_columns, device=device)
        block_list.append(torch.zeros((nb_samples, nb_columns, remaining_rows), device=device))

    final_matrix = torch.cat(block_list, dim=2).type(dtype)
    final_matrix = F.normalize(final_matrix, p=2, dim=2)

    if random_shuffle:
        _, indices = torch.rand((final_matrix.shape[0], final_matrix.shape[2]), device=device).sort(dim=1)
        indices = indices.unsqueeze(1).expand(final_matrix.shape)
        final_matrix = torch.gather(final_matrix, dim=2, index=indices)

    return final_matrix  # (nb_samples, nb_columns, nb_rows)


if __name__ == "__main__":
    gaussian_orthogonal_random_matrix(37, 128)
    gaussian_orthogonal_random_matrix_batched(256, 37, 128)
