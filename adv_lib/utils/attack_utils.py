import warnings
from collections import OrderedDict
from distutils.version import LooseVersion
from functools import partial
from inspect import isclass
from typing import Callable, Optional, Dict, Union

import numpy as np
import torch
import tqdm
from torch import Tensor, nn
from torch.nn import functional as F

from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from adv_lib.utils import ForwardCounter, BackwardCounter, predict_inputs


def generate_random_targets(labels: Tensor, num_classes: int) -> Tensor:
    """
    Generates one random target in (num_classes - 1) possibilities for each label that is different from the original
    label.

    Parameters
    ----------
    labels: Tensor
        Original labels. Generated targets will be different from labels.
    num_classes: int
        Number of classes to generate the random targets from.

    Returns
    -------
    targets: Tensor
        Random target for each label. Has the same shape as labels.

    """
    random = torch.rand(len(labels), num_classes, device=labels.device, dtype=torch.float)
    random.scatter_(1, labels.unsqueeze(-1), 0)
    return random.argmax(1)


def get_all_targets(labels: Tensor, num_classes: int):
    """
    Generates all possible targets that are different from the original labels.

    Parameters
    ----------
    labels: Tensor
        Original labels. Generated targets will be different from labels.
    num_classes: int
        Number of classes to generate the random targets from.

    Returns
    -------
    targets: Tensor
        Random targets for each label. shape: (len(labels), num_classes - 1).

    """
    all_possible_targets = torch.zeros(len(labels), num_classes - 1, dtype=torch.long)
    all_classes = set(range(num_classes))
    for i in range(len(labels)):
        this_label = labels[i].item()
        other_labels = list(all_classes.difference({this_label}))
        all_possible_targets[i] = torch.tensor(other_labels)

    return all_possible_targets


def run_attack(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               attack: Callable,
               targets: Optional[Tensor] = None,
               batch_size: Optional[int] = None) -> dict:
    device = next(model.parameters()).device
    to_device = lambda tensor: tensor.to(device)
    targeted, adv_labels = False, labels
    if targets is not None:
        targeted, adv_labels = True, targets
    batch_size = batch_size or len(inputs)

    # run attack only on non already adversarial samples
    already_adv = []
    chunks = [tensor.split(batch_size) for tensor in [inputs, adv_labels]]
    for (inputs_chunk, label_chunk) in zip(*chunks):
        batch_chunk_d, label_chunk_d = [to_device(tensor) for tensor in [inputs_chunk, label_chunk]]
        preds = model(batch_chunk_d).argmax(1)
        is_adv = (preds == label_chunk_d) if targeted else (preds != label_chunk_d)
        already_adv.append(is_adv.cpu())
    not_adv = ~torch.cat(already_adv, 0)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)
    average_forwards, average_backwards = [], []  # number of forward and backward calls per sample
    advs_chunks = []
    chunks = [tensor.split(batch_size) for tensor in [inputs[not_adv], adv_labels[not_adv]]]
    total_time = 0
    for (inputs_chunk, label_chunk) in tqdm.tqdm(zip(*chunks), ncols=80, total=len(chunks[0])):
        batch_chunk_d, label_chunk_d = [to_device(tensor) for tensor in [inputs_chunk, label_chunk]]

        start.record()
        advs_chunk_d = attack(model, batch_chunk_d, label_chunk_d, targeted=targeted)

        # performance monitoring
        end.record()
        torch.cuda.synchronize()
        total_time += (start.elapsed_time(end)) / 1000  # times for cuda Events are in milliseconds
        average_forwards.append(forward_counter.num_samples_called / len(batch_chunk_d))
        average_backwards.append(backward_counter.num_samples_called / len(batch_chunk_d))
        forward_counter.reset(), backward_counter.reset()

        advs_chunks.append(advs_chunk_d.cpu())
        if isinstance(attack, partial) and (callback := attack.keywords.get('callback')) is not None:
            callback.reset_windows()

    adv_inputs = inputs.clone()
    adv_inputs[not_adv] = torch.cat(advs_chunks, 0)

    data = {
        'inputs': inputs,
        'labels': labels,
        'targets': adv_labels if targeted else None,
        'adv_inputs': adv_inputs,
        'time': total_time,
        'num_forwards': sum(average_forwards) / len(chunks[0]),
        'num_backwards': sum(average_backwards) / len(chunks[0]),
    }

    return data


_default_metrics = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


def compute_attack_metrics(model: nn.Module,
                           attack_data: Dict[str, Union[Tensor, float]],
                           batch_size: Optional[int] = None,
                           metrics: Dict[str, Callable] = _default_metrics) -> Dict[str, Union[Tensor, float]]:
    inputs, labels, targets, adv_inputs = map(attack_data.get, ['inputs', 'labels', 'targets', 'adv_inputs'])
    if adv_inputs.min() < 0 or adv_inputs.max() > 1:
        warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
        adv_inputs.clamp_(min=0, max=1)
    device = next(model.parameters()).device
    to_device = lambda tensor: tensor.to(device)

    batch_size = batch_size or len(inputs)
    chunks = [tensor.split(batch_size) for tensor in [inputs, labels, adv_inputs]]
    all_predictions = [[] for _ in range(6)]
    distances = {k: [] for k in metrics.keys()}
    metrics = {k: v().to(device) if (isclass(v.func) if isinstance(v, partial) else False) else v for k, v in
               metrics.items()}

    append = lambda list, data: list.append(data.cpu())
    for inputs_chunk, labels_chunk, adv_chunk in zip(*chunks):
        inputs_chunk, adv_chunk = map(to_device, [inputs_chunk, adv_chunk])
        clean_preds, adv_preds = [predict_inputs(model, chunk.to(device)) for chunk in [inputs_chunk, adv_chunk]]
        list(map(append, all_predictions, [*clean_preds, *adv_preds]))
        for metric, metric_func in metrics.items():
            distances[metric].append(metric_func(adv_chunk, inputs_chunk).detach().cpu())

    logits, probs, preds, logits_adv, probs_adv, preds_adv = [torch.cat(l) for l in all_predictions]
    for metric in metrics.keys():
        distances[metric] = torch.cat(distances[metric], 0)

    accuracy_orig = (preds == labels).float().mean().item()
    if targets is not None:
        success = (preds_adv == targets)
        labels = targets
    else:
        success = (preds_adv != labels)

    prob_orig = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    prob_adv = probs_adv.gather(1, labels.unsqueeze(1)).squeeze(1)
    labels_infhot = torch.zeros_like(logits_adv).scatter_(1, labels.unsqueeze(1), float('inf'))
    real = logits_adv.gather(1, labels.unsqueeze(1)).squeeze(1)
    other = (logits_adv - labels_infhot).max(1).values
    diff_vs_max_adv = (real - other)
    nll = F.cross_entropy(logits, labels, reduction='none')
    nll_adv = F.cross_entropy(logits_adv, labels, reduction='none')

    data = {
        'time': attack_data['time'],
        'num_forwards': attack_data['num_forwards'],
        'num_backwards': attack_data['num_backwards'],
        'targeted': targets is not None,
        'preds': preds,
        'adv_preds': preds_adv,
        'accuracy_orig': accuracy_orig,
        'success': success,
        'probs_orig': prob_orig,
        'probs_adv': prob_adv,
        'logit_diff_adv': diff_vs_max_adv,
        'nll': nll,
        'nll_adv': nll_adv,
        'distances': distances,
    }

    return data


def print_metrics(metrics: dict) -> None:
    np.set_printoptions(formatter={'float': '{:0.3f}'.format}, threshold=16, edgeitems=3,
                        linewidth=120)  # To print arrays with less precision
    print('Original accuracy: {:.2%}'.format(metrics['accuracy_orig']))
    print('Attack done in: {:.2f}s with {:.4g} forwards and {:.4g} backwards.'.format(
        metrics['time'], metrics['num_forwards'], metrics['num_backwards']))
    success = metrics['success'].numpy()
    fail = success.mean() != 1
    print('Attack success: {:.2%}'.format(success.mean()) + fail * ' - {}'.format(success))
    for distance, values in metrics['distances'].items():
        data = values.numpy()
        print('{}: {} - Average: {:.3f} - Median: {:.3f}'.format(distance, data, data.mean(), np.median(data)) +
              fail * ' | Avg over success: {:.3f}'.format(data[success].mean()))
    attack_type = 'targets' if metrics['targeted'] else 'correct'
    print('Logit({} class) - max_Logit(other classes): {} - Average: {:.2f}'.format(
        attack_type, metrics['logit_diff_adv'].numpy(), metrics['logit_diff_adv'].numpy().mean()))
    print('NLL of target/pred class: {:.3f}'.format(metrics['nll_adv'].numpy().mean()))
