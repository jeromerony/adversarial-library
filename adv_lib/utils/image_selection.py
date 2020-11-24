from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


def select_images(model: nn.Module, dataset: Dataset, num_images: int, correct_only: bool = False,
                  random: bool = False) -> Tuple[Tensor, Tensor]:
    device = next(model.parameters()).device
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=random)
    selected_images, selected_labels = [], []
    for (image, label) in loader:
        if correct_only:
            correct = model(image.to(device)).argmax(1) == label.to(device)
            if correct.all():
                selected_images.append(image), selected_labels.append(label)
        elif not correct_only:
            selected_images.append(image), selected_labels.append(label)

        if len(selected_images) == num_images:
            break
    else:
        print('Could only find {} correctly classified images.'.format(len(selected_images)))

    return torch.cat(selected_images, 0), torch.cat(selected_labels, 0)
