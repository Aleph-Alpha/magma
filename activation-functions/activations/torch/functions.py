import torch
import torch.nn.functional as F
from activations.utils.find_init_weights import find_weights
from activations.utils.utils import _get_auto_axis_layout, _cleared_arrays
from activations.utils.warnings import RationalImportScipyWarning
from activations.utils.activation_logger import ActivationLogger
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from termcolor import colored
from random import randint


_LINED = dict()


def create_colors(n):
    colors = []
    for i in range(n):
        colors.append('#%06X' % randint(0, 0xFFFFFF))
    return colors


def _save_inputs(self, input, output):
    if self._selected_distribution is None:
        raise ValueError("Selected distribution is none")
    self._selected_distribution.fill_n(input[0])


def _save_gradients(self, in_grad, out_grad):
    self._in_grad_dist.fill_n(in_grad[0])
    self._out_grad_dist.fill_n(out_grad[0])


def _save_inputs_auto_stop(self, input, output):
    self.inputs_saved += 1
    if self._selected_distribution is None:
        raise ValueError("Selected distribution is none")
    self._selected_distribution.fill_n(input[0])
    if self.inputs_saved > self._max_saves:
        self.training_mode()


class Metaclass(type):
    def __setattr__(self, key, value):
        if not hasattr(self, key):
            key_str = colored(key, "red")
            self_name_str = colored(self, "red")
            msg = colored(f"Setting new Class attribute {key_str}", "yellow") + \
                  colored(f" of {self_name_str}", "yellow")
            print(msg)
        type.__setattr__(self, key, value)


class ActivationModule(torch.nn.Module):#, metaclass=Metaclass):
    # histograms_colors = plt.get_cmap('Pastel1').colors
    instances = {}
    histograms_colors = ["red", "green", "black"]
    distribution_display_mode = "kde"

    def __init__(self, function, device=None):
        if isinstance(function, str):
            self.type = function
            function = None
        super().__init__()
        self.logger = ActivationLogger(f"ActivationLogger: {function}")
        if self.classname not in self.instances:
            self.instances[self.classname] = []
        self.instances[self.classname].append(self)
        if function is not None:
            self.activation_function = function
            if "__forward__" in dir(function):
                self.forward = self.activation_function.__forward__
            else:
                self.forward = self.activation_function
        self._handle_inputs = None
        self._handle_grads = None
        self._saving_input = False
        self.distributions = []
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.use_kde = True

    @property
    def classname(self):
        clsn = str(self.__class__)
        if "activations.torch" in clsn:
            return clsn.split("'")[1].split(".")[-1]
        else:
            return "Unknown"  # TODO, implement

    def save_inputs(self, saving=True, auto_stop=False, max_saves=1000,
                    bin_width=0.1, mode="all", category_name=None):
        """
        Will retrieve the distribution of the input in self.distribution. \n
        This will slow down the function, as it has to retrieve the input \
        dist.\n

        Arguments:
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``all``
                category_name (str):
                    The name of the category
                    Default ``0``
        """
        if not saving:
            self.logger.warn("Not retrieving input anymore")
            self._handle_inputs.remove()
            self._handle_inputs = None
            return
        if self._handle_inputs is not None:
            # print("Already in retrieve mode")
            return
        if "cuda" in self.device:
            if "neurons" in mode.lower():
                from activations.torch.utils.histograms_cupy import NeuronsHistogram as Histogram
            else:
                from activations.torch.utils.histograms_cupy import Histogram
        else:
            if "neurons" in mode.lower():
                from activations.torch.utils.histograms_numpy import NeuronsHistogram as Histogram
            else:
                from activations.torch.utils.histograms_numpy import Histogram
        if "categor" in mode.lower():
            if category_name is None:
                self._selected_distribution_name = None
                self.categories = []
                self._selected_distribution = None
                self.distributions = []
            else:
                self._selected_distribution_name = category_name
                self.categories = [category_name]
                self._selected_distribution = Histogram(bin_width)
                self.distributions = [self._selected_distribution]
        else:
            self._selected_distribution_name = "distribution"
            self.categories = ["distribution"]
            self._selected_distribution = Histogram(bin_width)
            self.distributions = [self._selected_distribution]
        self._irm = mode  # input retrieval mode
        self._inp_bin_width = bin_width
        if auto_stop:
            self.inputs_saved = 0
            self._handle_inputs = self.register_forward_hook(_save_inputs_auto_stop)
            self._max_saves = max_saves
        else:
            self._handle_inputs = self.register_forward_hook(_save_inputs)

    def save_gradients(self, saving=True, auto_stop=False, max_saves=1000,
                       bin_width="auto", mode="all"):
        """
        Will retrieve the distribution of the input in self.distribution. \n
        This will slow down the function, as it has to retrieve the input \
        dist.\n

        Arguments:
                auto_stop (bool):
                    If True, the retrieving will stop after `max_saves` \
                    calls to forward.\n
                    Else, use :meth:`torch.Rational.training_mode`.\n
                    Default ``False``
                max_saves (int):
                    The range on which the curves of the functions are fitted \
                    together.\n
                    Default ``1000``
                bin_width (float):
                    Default bin width for the histogram.\n
                    Default ``0.1``
                mode (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``all``
                category_name (str):
                    The mode for the input retrieve.\n
                    Have to be one of ``all``, ``categories``, ...
                    Default ``0``
        """
        if not saving:
            self.logger.warn("Not retrieving gradients anymore")
            self._handle_grads.remove()
            self._handle_grads = None
            return
        if self._handle_grads is not None:
            # print("Already in retrieve mode")
            return
        if "cuda" in self.device:
            from .utils.histograms_cupy import Histogram
        else:
            from .utils.histograms_numpy import Histogram

        self._grm = mode  # gradient retrieval mode
        self._in_grad_dist = Histogram(bin_width)
        self._out_grad_dist = Histogram(bin_width)
        self._grad_bin_width = bin_width
        if auto_stop:
            self.inputs_saved = 0
            raise NotImplementedError
            # self._handle_grads = self.register_full_backward_hook(_save_gradients_auto_stop)
            self._max_saves = max_saves
        else:
            self._handle_grads = self.register_full_backward_hook(_save_gradients)

    # def training_mode(self):
    #     """
    #     Stops retrieving the distribution of the input in `self.distribution`.
    #     """
    #     print("Training mode, no longer retrieving the input.")
    #     self._handle_inputs.remove()
    #     self._handle_inputs = None

    @classmethod
    def save_all_inputs(cls, *args, **kwargs):
        """
        Saves inputs for all instantiates objects of the called class.
        """
        instances_list = cls._get_instances()
        for instance in instances_list:
            instance.save_inputs(*args, **kwargs)

    @classmethod
    def save_all_gradients(cls, *args, **kwargs):
        """
        Saves gradients for all instantiates objects of the called class.
        """
        instances_list = cls._get_instances()
        for instance in instances_list:
            instance.save_gradients(*args, **kwargs)

    def __repr__(self):
        return f"{self.classname}"
        # if "type" in dir(self):
        #     # return  f"{self.type} ActivationModule at {hex(id(self))}"
        #     return  f"{self.type} ActivationModule"
        # if "__name__" in dir(self.activation_function):
        #     # return f"{self.activation_function.__name__} ActivationModule at {hex(id(self))}"
        #     return f"{self.activation_function.__name__} ActivationModule"
        # return f"{self.activation_function} ActivationModule"

    def show_gradients(self, display=True, tolerance=0.001, title=None,
                       axis=None, writer=None, step=None, label=None, colors=None):
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False
        if axis is None:
            with sns.axes_style("whitegrid"):
                # fig, axis = plt.subplots(1, 1, figsize=(8, 6))
                fig, axis = plt.subplots(1, 1, figsize=(20, 12))
        if colors is None or len(colors) != 2:
            colors = ["orange", "blue"]
        dists = [self._in_grad_dist, self._out_grad_dist]
        if label is None:
            labels = ['input grads', 'output grads']
        else:
            labels = [f'{label} (inp)', f'{label} (outp)']
        for distribution, col, label in zip(dists, colors, labels):
            weights, x = distribution.weights, distribution.bins
            if self.use_kde and scipy_imported:
                if len(x) > 5:
                    refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                    kde_curv = distribution.kde()(refined_bins)
                    # ax.plot(refined_bins, kde_curv, lw=0.1)
                    axis.fill_between(refined_bins, kde_curv, alpha=0.4,
                                      color=col, label=label)
                else:
                    self.logger.warn("The bin size is too big, bins contain too few "
                                     f"elements.\nbins: {x}")
                    axis.bar([], []) # in case of remove needed
            else:
                axis.bar(x, weights/weights.max(), width=x[1] - x[0],
                         linewidth=0, alpha=0.4, color=col, label=label)
            distribution.empty()
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            plt.legend()
            plt.show()
        else:
            if axis is None:
                return fig

    @classmethod
    def show_all_gradients(cls, display=True, tolerance=0.001, title=None,
                           axes=None, layout="auto", writer=None, step=None,
                           colors=None):
        """
        Shows a graph of the all instanciated activation functions (or returns \
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
        logger = ActivationLogger("f{cls.__name__}Logger")
        instances_list = cls._get_instances()
        if axes is None:
            if layout == "auto":
                total = len(instances_list)
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
                logger.warn("Could not import seaborn")
                #RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(instances_list):]:
                ax.remove()
            axes = axes[:len(instances_list)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(instances_list))])
            fig = plt.gcf()
        if isinstance(colors, str) or colors is None:
            colors = [colors]*len(axes.flatten())
        for act, ax, color in zip(instances_list, axes.flatten(), colors):
            act.show_gradients(False, tolerance, title, axis=ax,
                               writer=None, step=step, colors=color)
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
            plt.legend()
            plt.show()
        else:
            return fig

    def show(self, x=None, fitted_function=True, other_func=None, display=True,
             tolerance=0.001, title=None, axis=None, writer=None, step=None, label=None,
             color=None):
        #Construct x axis
        if x is None:
            x = torch.arange(-3., 3, 0.01)
        elif isinstance(x, tuple) and len(x) in (2, 3):
            x = torch.arange(*x).float()
        elif isinstance(x, torch.Tensor) or isinstance(x, np.ndarray):
            x = torch.tensor(x.float())
        if axis is None:
            with sns.axes_style("whitegrid"):
                # fig, axis = plt.subplots(1, 1, figsize=(8, 6))
                fig, axis = plt.subplots(1, 1, figsize=(20, 12))
        if self.distributions:
            if self.distribution_display_mode in ["kde", "bar"]:
                ax2 = axis.twinx()
                if "neurons" in self._irm:
                    x = self.plot_layer_distributions(ax2)
                else:
                    x = self.plot_distributions(ax2, color)
            elif self.distribution_display_mode == "points":
                x0, x_last, _ = self.get_distributions_range()
                x_edges = torch.tensor([x0, x_last]).float()
                y_edges = self.forward(x_edges.to(self.device)).detach().cpu().numpy()
                axis.scatter(x_edges, y_edges, color=color)
        #TODO: this should enable showing without input data from before
        y = self.forward(x.to(self.device)).detach().cpu().numpy()
        if label:
            # axis.twinx().plot(x, y, label=label, color=color)
            axis.plot(x, y, label=label, color=color)
        else:
            # axis.twinx().plot(x, y, label=label, color=color)
            axis.plot(x, y, label=label, color=color)
        if writer is not None:
            try:
                writer.add_figure(title, fig, step)
            except AttributeError:
                self.logger.error("Could not use the given SummaryWriter to add the Rational figure")
        elif display:
            plt.show()
        else:
            if axis is None:
                return fig

    @property
    def current_inp_category(self):
        return self._selected_distribution_name

    @current_inp_category.setter
    def current_inp_category(self, value):
        if value == self._selected_distribution_name:
            return
        if "cuda" in self.device:
            if "neurons" in self._irm.lower():
                from activations.torch.utils.histograms_cupy import NeuronsHistogram as Histogram
            else:
                from activations.torch.utils.histograms_cupy import Histogram
        else:
            if "neurons" in self._irm.lower():
                from activations.torch.utils.histograms_numpy import NeuronsHistogram as Histogram
            else:
                from activations.torch.utils.histograms_numpy import Histogram
        #if the histogram is empty, it means that is was created at the same phase
        #that the current category is created, which means that no input was perceived
        #during this time -> redundant category
        for i in range(len(self.distributions)):
            if self.distributions[i].is_empty:
                del self.distributions[i]
                del self.categories[i]
        self._selected_distribution = Histogram(self._inp_bin_width)
        self.distributions.append(self._selected_distribution)
        self.categories.append(value)
        self._selected_distribution_name = value

    def plot_distributions(self, ax, colors=None, bin_size=None):
        """
        Plot the distribution and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
            scipy_imported = False
        dists_fb = []
        x_min, x_max = np.inf, -np.inf
        #TODO: this is obsolete afaik
        """ if colors is None:
            colors = self.histograms_colors """
        if not(isinstance(colors, list) or isinstance(colors, tuple)):
            colors = create_colors(len(self.distributions))
        for i, (distribution, inp_label, color) in enumerate(zip(self.distributions, self.categories, colors)):
            if distribution.is_empty:
                if self.distribution_display_mode == "kde" and scipy_imported:
                    fill = ax.fill_between([], [], label=inp_label,  alpha=0.)
                else:
                    fill = ax.bar([], [], label=inp_label,  alpha=0.)
                dists_fb.append(fill)
            else:
                weights, x = _cleared_arrays(distribution.weights, distribution.bins, 0.001)
                # weights, x = distribution.weights, distribution.bins
                if self.distribution_display_mode == "kde" and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(x[0], x[-1], 200)
                        kde_curv = distribution.kde()(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.45,
                                               color=color, label=inp_label)
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                        fill = ax.bar([], []) # in case of remove needed
                    size = x[1] - x[0]
                else:
                    width = (x[1] - x[0])/len(self.distributions)
                    if len(x) == len(weights):
                        fill = ax.bar(x+i*width, weights/weights.max(), width=width,
                                  linewidth=0, alpha=0.7, label=inp_label)
                    else:
                        fill = ax.bar(x[1:]+i*width, weights/weights.max(), width=width,
                                  linewidth=0, alpha=0.7, label=inp_label)
                    size = (x[1] - x[0])/100 # bar size can be larger
                dists_fb.append(fill)
                x_min, x_max = min(x_min, x[0]), max(x_max, x[-1])
        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.4)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                _LINED[legline] = origline
            fig = plt.gcf()
            def toggle_distribution(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = _LINED[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()[0]
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(0.4)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_distribution)
        if x_min == np.inf or x_max == np.inf:
            torch.arange(-3, 3, 0.01)
        #TODO: when distribution is always empty, size wont be assigned and will throw an error

        return torch.arange(x_min, x_max, size)

    def plot_layer_distributions(self, ax):
        """
        Plot the layer distributions and returns the corresponding x
        """
        ax.set_yticks([])
        try:
            import scipy.stats as sts
            scipy_imported = True
        except ImportError:
            RationalImportScipyWarning.warn()
        dists_fb = []
        for distribution, inp_label, color in zip(self.distributions, self.categories, self.histograms_colors):
            #TODO: why is there no empty distribution check here?
            for n, (weights, x) in enumerate(zip(distribution.weights, distribution.bins)):
                if self.use_kde and scipy_imported:
                    if len(x) > 5:
                        refined_bins = np.linspace(float(x[0]), float(x[-1]), 200)
                        kde_curv = distribution.kde(n)(refined_bins)
                        # ax.plot(refined_bins, kde_curv, lw=0.1)
                        fill = ax.fill_between(refined_bins, kde_curv, alpha=0.4,
                                                color=color, label=f"{inp_label} ({n})")
                    else:
                        self.logger.warn(f"The bin size is too big, bins contain too few "
                              "elements.\nbins: {x}")
                        fill = ax.bar([], []) # in case of remove needed
                else:
                    fill = ax.bar(x, weights/weights.max(), width=x[1] - x[0],
                                  linewidth=0, alpha=0.4, color=color,
                                  label=f"{inp_label} ({n})")
                dists_fb.append(fill)

        if self.distribution_display_mode in ["kde", "bar"]:
            leg = ax.legend(fancybox=True, shadow=True)
            leg.get_frame().set_alpha(0.4)
            for legline, origline in zip(leg.get_patches(), dists_fb):
                legline.set_picker(5)  # 5 pts tolerance
                _LINED[legline] = origline
            fig = plt.gcf()
            def toggle_distribution(event):
                # on the pick event, find the orig line corresponding to the
                # legend proxy line, and toggle the visibility
                leg = event.artist
                orig = _LINED[leg]
                if "get_visible" in dir(orig):
                    vis = not orig.get_visible()
                    orig.set_visible(vis)
                    color = orig.get_facecolors()[0]
                else:
                    vis = not orig.patches[0].get_visible()
                    color = orig.patches[0].get_facecolor()
                    for p in orig.patches:
                        p.set_visible(vis)
                # Change the alpha on the line in the legend so we can see what lines
                # have been toggled
                if vis:
                    leg.set_alpha(0.4)
                else:
                    leg.set_alpha(0.)
                leg.set_facecolor(color)
                fig.canvas.draw()
            fig.canvas.mpl_connect('pick_event', toggle_distribution)
        return torch.arange(*self.get_distributions_range())

    def get_distributions_range(self):
        x_min, x_max = np.inf, -np.inf
        for dist in self.distributions:
            if not dist.is_empty:
                x_min, x_max = min(x_min, dist.range[0]), max(x_max, dist.range[-1])
                size = dist.range[1] - dist.range[0]
        if x_min == np.inf or x_max == np.inf:
            return -3, 3, 0.01
        return x_min, x_max, size

    @classmethod
    def _get_instances(cls):
        """
        if called from ActivationModule: returning all instanciated functions
        if called from a child-class: returning the instances of this specific class
        """
        if "ActivationModule" in str(cls):
            instances_list = []
            [instances_list.extend(insts) for insts in cls.instances.values()]
        else:
            clsn = str(cls)
            if "activations.torch" in clsn:
                curr_classname = clsn.split("'")[1].split(".")[-1]
                if curr_classname not in cls.instances:
                    print(f"No instanciated function of {curr_classname} found")
                    return []
                instances_list = cls.instances[curr_classname]
            else:
                print(f"Unknown {cls} for show_all")  # shall never happen
                return []
        return instances_list

    @classmethod
    def show_all(cls, x=None, fitted_function=True, other_func=None,
                 display=True, tolerance=0.001, title=None, axes=None,
                 layout="auto", writer=None, step=None, colors="#1f77b4"):
        """
        Shows a graph of the all instanciated activation functions (or returns \
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
        logger = ActivationLogger(f"{cls.__name__}Logger")
        instances_list = cls._get_instances()
        if axes is None:
            if layout == "auto":
                total = len(instances_list)
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
                logger.warn("Could not import seaborn")
                #RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(instances_list):]:
                ax.remove()
            axes = axes[:len(instances_list)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(instances_list))])
            fig = plt.gcf()
        if isinstance(colors, str):
            colors = [colors]*len(axes.flatten())
        if isinstance(x, list):
            for act, ax, x_act, color in zip(instances_list, axes.flatten(), x, colors):
                act.show(x_act, fitted_function, other_func, False, tolerance,
                         title, axis=ax, writer=None, step=step,
                         color=color)
        else:
            for act, ax, color in zip(instances_list, axes.flatten(), colors):
                act.show(x, fitted_function, other_func, False, tolerance,
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

    # def __setattr__(self, key, value):
    #     if not hasattr(self, key):
    #         key_str = colored(key, "red")
    #         self_name_str = colored(self.__class__, "red")
    #         msg = colored(f"Setting new attribute {key_str}", "yellow") + \
    #               colored(f" of instance of {self_name_str}", "yellow")
    #         print(msg)
    #     object.__setattr__(self, key, value)


    # def load_state_dict(self, state_dict):
    #     if "distributions" in state_dict.keys():
    #         _distributions = state_dict.pop("distributions")
    #         if "cuda" in self.device and _cupy_installed():
    #             msg = f"Loading input distributions on {self.device} using cupy"
    #             RationalLoadWarning.warn(msg)
    #             self.distributions = _distributions
    #     super().load_state_dict(state_dict)
    #
    def state_dict(self, destination=None, *args, **kwargs):
        _state_dict = super().state_dict(destination, *args, **kwargs)
        if self.distributions is not None:
            _state_dict["distributions"] = self.distributions
        return _state_dict



if __name__ == '__main__':
    def plot_gaussian(mode, device):
        _2pi_sqrt = 2.5066
        tanh = torch.tanh
        relu = F.relu

        nb_neurons_in_layer = 5

        leaky_relu = F.leaky_relu
        gaussian = lambda x: torch.exp(-0.5*x**2) / _2pi_sqrt
        gaussian.__name__ = "gaussian"
        gau = ActivationModule(gaussian, device=device)
        gau.save_inputs(mode=mode, category_name="neg") # Wrong
        inp = torch.stack([(torch.rand(10000)-(i+1))*2 for i in range(nb_neurons_in_layer)], 1)
        print(inp.shape)
        gau(inp.to(device))
        if "categories" in mode:
            gau.current_inp_category = "pos"
            inp = torch.stack([(torch.rand(10000)+(i+1))*2 for i in range(nb_neurons_in_layer)], 1)
            gau(inp.to(device))
            # gau(inp.cuda())
        gau.show()

    ActivationModule.distribution_display_mode = "bar"
    # for device in ["cuda:0", "cpu"]:
    for device in ["cpu"]:
        for mode in ["categories", "neurons", "neurons_categories"]:
            plot_gaussian(mode, device)
