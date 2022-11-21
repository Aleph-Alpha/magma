from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from .functions import ActivationModule
from activations.utils.utils import _get_auto_axis_layout


def tent_activation(x, delta):
    """
    Functional implementation of TentActivation.
    """
    return torch.clamp(delta - torch.abs(x), min=0)


class TentActivation(ActivationModule):
    distribution_display_mode = "kde"
    list = []

    def __init__(self, delta: Union[torch.Tensor, float] = 2.0, lb=0.0, ub=500.0, learnable: bool = False):
        """
        Applies element-wise Tent(x) = max(0, delta - |x|)
        :param delta: The delta which is used as initialization
        :param lb:  The lower bound of the possible delta values
        :param ub:  The upper bound of the possible delta values
        """
        super().__init__("tent")
        if torch.is_tensor(delta):
            self.delta = nn.Parameter(delta, requires_grad=learnable)
        else:
            self.delta = nn.Parameter(torch.tensor(delta), requires_grad=learnable)
        # self.delta.requires_grad = learnable
        self.lb = lb
        self.ub = ub
        self.learnable = learnable
        self.list.append(self)

    def forward(self, x: Tensor) -> Tensor:
        return tent_activation(x, self.delta)

    def extra_repr(self) -> str:
        return f'delta={self.delta}, lb={self.lb}, ub={self.ub}, learnable={self.learnable}'

    def __str__(self):
        return "BiTent"

    @classmethod
    def show_all(cls, x=None, fitted_function=True, other_func=None,
                 display=True, tolerance=0.001, title=None, axes=None,
                 layout="auto", writer=None, step=None, colors="#1f77b4"):
        """
        Shows a graph of the all instanciated rational functions (or returns \
        it if ``returns=True``).

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_funcs (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                display (bool):
                    If ``True``, displays the plot.
                    Otherwise, returns the figure. \n
                    Default ``False``
                tolerance (float):
                    If the input histogram is used, it will be pruned. \n
                    Every bin containg less than `tolerance` of the total \
                    input is pruned out.
                    (Reduces noise).
                    Default ``0.001``
                title (str):
                    If not None, a title for the figure
                    Default ``None``
                axes (matplotlib.pyplot.axis):
                    On ax or a list of axes to be plotted on. \n
                    If None, creates them automatically (see `layout`). \n
                    Default ``None``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\n
                    Default ``"auto"``
                writer (tensorboardX.SummaryWriter):
                    A tensorboardX writer to give the image to, in case of
                    debugging.
                    Default ``None``
                step (int):
                    A step/epoch for tensorboardX writer.
                    If None, incrementing itself.
                    Default ``None``
        """
        if axes is None:
            if layout == "auto":
                total = len(cls.list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            figs = tuple(np.flip(np.array(layout)* (2, 3)))
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figs)
            except ImportError:
                RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(cls.list):]:
                ax.remove()
            axes = axes[:len(cls.list)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(cls.list))])
            fig = plt.gcf()
        if isinstance(colors, str):
            colors = [colors]*len(axes.flatten())
        if isinstance(x, list):
            for rat, ax, x_rat, color in zip(cls.list, axes.flatten(), x, colors):
                rat.show(x_rat, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        else:
            for rat, ax, color in zip(cls.list, axes.flatten(), colors):
                rat.show(x, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        if title is not None:
            fig.suptitle(title, y=0.95)
        fig = plt.gcf()
        fig.tight_layout()
        if writer is not None:
            if step is None:
                step = cls._step
                cls._step += 1
            writer.add_figure(title, fig, step)
        elif display:
            # plt.legend()
            plt.show()
        else:
            return fig


def bitent_activation(x, delta, epsilon):
    """
    Functional implementation of BiTentActivation.
    """
    hdt = delta/2
    return tent_activation(x+hdt+epsilon, hdt) + tent_activation(x-hdt-epsilon, hdt)


class BiTentActivation(TentActivation):
    def __init__(self, delta: Union[torch.Tensor, float] = 2.0, lb=0.0, ub=500.0, learnable: bool = False):
        super().__init__(delta, lb, ub, learnable)
        self.epsilon = 0.1 / 2

    def forward(self, x: Tensor) -> Tensor:
        return bitent_activation(x, self.delta, self.epsilon)
