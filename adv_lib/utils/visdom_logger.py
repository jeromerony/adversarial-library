from collections import defaultdict
from enum import Enum
from typing import List, Optional, Tuple, Union

import torch
import visdom
from torch import Tensor


class ChartTypes(Enum):
    line = 1,
    image = 2


class ChartData:
    def __init__(self):
        self.window = None
        self.type = None
        self.x_list = []
        self.y_list = []
        self.other_data = None
        self.to_plot = {}


class VisdomLogger:
    def __init__(self, port: int):
        self.vis = visdom.Visdom(port=port)
        self.windows = defaultdict(lambda: ChartData())

    @staticmethod
    def as_unsqueezed_tensor(data: Union[float, List[float], Tensor]) -> Tensor:
        data = torch.as_tensor(data).detach()
        return data.unsqueeze(0) if data.ndim == 0 else data

    def accumulate_line(self, names: Union[str, List[str]], x: Union[float, Tensor],
                        y: Union[float, Tensor, List[Tensor]], title: str = '', **kwargs) -> None:
        if isinstance(names, str):
            names = [names]
        data = self.windows['$'.join(names)]
        update = None if data.window is None else 'append'

        if isinstance(y, (int, float)):
            Y = torch.tensor([y])
        elif isinstance(y, list):
            Y = torch.stack(list(map(self.as_unsqueezed_tensor, y)), 1)
        elif isinstance(y, Tensor):
            Y = self.as_unsqueezed_tensor(y)

        if isinstance(x, (int, float)):
            X = torch.tensor([x])
        elif isinstance(x, Tensor):
            X = self.as_unsqueezed_tensor(x)

        if Y.ndim == 2 and X.ndim == 1:
            X.expand(len(X), Y.shape[1])

        if len(data.to_plot) == 0:
            data.to_plot = {'X': X, 'Y': Y, 'win': data.window, 'update': update,
                            'opts': {'legend': names, 'title': title, **kwargs}}
        else:
            data.to_plot['X'] = torch.cat((data.to_plot['X'], X), 0)
            data.to_plot['Y'] = torch.cat((data.to_plot['Y'], Y), 0)

    def update_lines(self) -> None:
        for window, data in self.windows.items():
            if len(data.to_plot) != 0:
                win = self.vis.line(**data.to_plot)

                data.x_list.append(data.to_plot['X'])
                data.y_list.append(data.to_plot['Y'])

                # Update the window
                data.window = win
                data.type = ChartTypes.line

                data.to_plot = {}

    def line(self, names: Union[str, List[str]], x: Union[float, Tensor], y: Union[float, Tensor, List[Tensor]],
             title: str = '', **kwargs) -> None:
        self.accumulate_line(names=names, x=x, y=y, title=title, **kwargs)
        self.update_lines()

    def images(self, name: str, images: Tensor, mean_std: Optional[Tuple[List[float], List[float]]] = None,
               title: str = '') -> None:
        data = self.windows[name]

        if mean_std is not None:
            images = images * torch.as_tensor(mean_std[0]) + torch.as_tensor(mean_std[1])

        win = self.vis.images(images, win=data.window, opts={'legend': [name], 'title': title})

        # Update the window
        data.window = win
        data.other_data = images
        data.type = ChartTypes.image

    def reset_windows(self):
        self.windows.clear()

    def save(self, filename):
        to_save = {}
        for (name, data) in self.windows.items():
            type = data.type
            if type == ChartTypes.line:
                to_save[name] = (type, torch.cat(data.x_list, dim=0).cpu(), torch.cat(data.y_list, dim=0).cpu())
            elif type == ChartTypes.image:
                to_save[name] = (type, data.other_data.cpu())

        torch.save(to_save, filename)
