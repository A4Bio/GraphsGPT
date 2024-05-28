import torch
from torch import no_grad


@no_grad()
def classification_accuracy(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
):
    prediction = torch.argmax(logits, dim=1)
    correct = (prediction == labels).float()
    return torch.mean(correct)
