"""
Rational Activation Functions (Base for every framework)
========================================================

This module allows you to create Rational Neural Networks using Learnable
Rational activation functions. This base function is used by Pytorch,
TensorFlow/Keras, and MXNET Rational Activation Functions.
"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from activations.utils.utils import Snapshot, _path_for_multiple, \
    _get_auto_axis_layout, _get_frontiers, _erase_suffix, _increment_string, \
    _repair_path, _cleared_arrays
from activations.utils.warnings import RationalWarning, \
    RationalImportSeabornWarning


class Rational_base():
    count = 0
    list = []
    # distribution_display_mode = "kde"
    use_multiple_axis = False
    _step = 0

    def __init__(self, name):
        self._handle_retrieve_mode = None
        self.distribution = None
        self.best_fitted_function = None
        self.best_fitted_function_params = None
        self.snapshot_list = list()
        self._verbose = True
        if name in [rat.func_name for rat in self.list]:
            name = _increment_string(name)
        self.func_name = name
        Rational_base.count += 1
        Rational_base.list.append(self)
        self._step = 0

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

    # def show(self, x=None, fitted_function=True, other_func=None, display=True,
    #          tolerance=0.001, title=None, axis=None, writer=None, step=None,
    #          color="#1f77b4"):
    #     """
    #     Shows a graph of the function (or returns it if ``returns=True``).
    #
    #     Arguments:
    #             x (range):
    #                 The range to print the function on.\n
    #                 Default ``None``
    #             fitted_function (bool):
    #                 If ``True``, displays the best fitted function if searched.
    #                 Otherwise, returns it. \n
    #                 Default ``True``
    #             other_funcs (callable):
    #                 another function to be plotted or a list of other callable
    #                 functions or a dictionary with the function name as key
    #                 and the callable as value.
    #             display (bool):
    #                 If ``True``, displays the plot.
    #                 Otherwise, returns the figure. \n
    #                 Default ``False``
    #             tolerance (float):
    #                 If the input histogram is used, it will be pruned. \n
    #                 Every bin containg less than `tolerance` of the total \
    #                 input is pruned out.
    #                 (Reduces noise).
    #                 Default ``0.001``
    #             title (str):
    #                 If not None, a title for the figure
    #                 Default ``None``
    #             axis (matplotlib.pyplot.axis):
    #                 axis to be plotted on. If None, creates one automatically.
    #                 Default ``None``
    #             writer (tensorboardX.SummaryWriter):
    #                 A tensorboardX writer to give the image to, in case of
    #                 debugging.
    #                 Default ``None``
    #             step (int):
    #                 A step/epoch for tensorboardX writer.
    #                 If None, incrementing itself.
    #                 Default ``None``
    #     """
    #     snap = self.capture(returns=True)
    #     # snap.histogram = self.distribution
    #     if title is None:
    #         rats_names = [_erase_suffix(rat.func_name) for rat in self.list]
    #         if len(set(rats_names)) != 1:
    #             title = self.func_name
    #     if axis is None:
    #         fig = snap.show(x, fitted_function, other_func, display, tolerance,
    #                         title, duplicate_axis=self.use_multiple_axis)
    #         if writer is not None:
    #             if step is None:
    #                 step = self._step
    #                 self._step += 1
    #             try:
    #                 writer.add_figure(title, fig, step)
    #             except AttributeError:
    #                 print("Could not use the given SummaryWriter to add the Rational figure")
    #         elif not display:
    #             return fig
    #     else:
    #         snap.show(x, fitted_function, other_func, display, tolerance,
    #                   title, axis=axis, duplicate_axis=self.use_multiple_axis,
    #                   color=color)

    @classmethod
    def capture_all(cls, name="snapshot_0", x=None, fitted_function=True,
                    other_func=None, returns=False):
        """
        Captures a snapshot of every instanciated rational functions and \
        related in the snapshot_list variable (or returns a list of them if \
        ``returns=True``).

        Arguments:
                name (str):
                    Name of the snapshot.\n
                    Default ``"snapshot_0"``
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
                returns (bool):
                    If ``True``, returns the snapshot.
                    Otherwise, saves it in self.snapshot_list \n
                    Default ``False``
        """
        if returns:
            captures = []
            for rat in cls.list:
                captures.append(rat.capture(name, x, fitted_function,
                                            other_func, returns))
            return captures
        else:
            for rat in cls.list:
                rat.capture(name, x, fitted_function, other_func, returns)

    def capture(self, name="snapshot_0", x=None, fitted_function=True,
                other_func=None, returns=False):
        """
        Captures a snapshot of the rational functions and related in the
        snapshot_list variable (or returns it if ``returns=True``).

        Arguments:
                name (str):
                    Name of the snapshot.\n
                    Default ``"snapshot_0"``
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
                returns (bool):
                    If ``True``, returns the snapshot.
                    Otherwise, saves it in self.snapshot_list \n
                    Default ``False``
        """
        while name in [snst.name for snst in self.snapshot_list] \
              and not returns:
            name = _increment_string(name)
        snapshot = Snapshot(name, self, fitted_function, other_func)
        if returns:
            return snapshot
        self.snapshot_list.append(snapshot)

    def export_graph(self, path="rational_function.svg", snap_number=-1,
                     other_func=None):
        """
        Saves one graph of the function based on the last snapshot \
        (by default, and if available).

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rational_functions.svg"``
                together (bool):
                    If True, the graphs of every functions are stored in \
                    different files.\n
                    Default ``True``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\
                    (see `layout`).
                    Default ``auto``
                snap_number (int):
                    The snap to take in snapshot_list for each function.\n
                    Default ``-1 (last)``
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                    Default ``None``
        """
        if not len(self.snapshot_list):
            mes =("Cannot use the last snapshot as the snapshot_list "
                  "is empty, making a capture with default params")
            RationalWarning.warn(mes)
            self.capture()
        snap = self.snapshot_list[snap_number]
        snap.save(path=path, other_func=other_func)

    @classmethod
    def export_graphs(cls, path="rational_functions.svg", together=True,
                      layout="auto", snap_number=-1, other_func=None):
        """
        Saves one or more graph(s) of the function based on the last snapshot \
        (by default, and if available) for each instanciated rational function.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rational_functions.svg"``
                together (bool):
                    If True, the graphs of every functions are stored in \
                    different files.\n
                    Default ``True``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\
                    (see `layout`).
                    Default ``"auto"``
                snap_number (int):
                    The snap to take in snapshot_list for each function.\n
                    Default ``-1 (last)``
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value.
                    Default ``None``
        """
        if together:
            for i, rat in enumerate(cls.list):
                if not len(rat.snapshot_list) > 0:
                    print(f"Cannot use the last snapshots as snapshot n {i} \
                          is empty, capturing...")
                    cls.capture_all()
                    break
            if layout == "auto":
                total = len(cls.list)
                layout = _get_auto_axis_layout(total)
            if len(layout) != 2:
                msg = 'layout should be either "auto" or a tuple of size 2'
                raise TypeError(msg)
            figs = tuple(np.flip(np.array(layout) * (2, 3)))
            try:
                import seaborn as sns
                with sns.axes_style("whitegrid"):
                    fig, axes = plt.subplots(*layout, figsize=figs)
            except ImportError:
                RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            for rat, ax in zip(cls.list, axes.flatten()):
                snap = rat.snapshot_list[snap_number]
                snap.show(display=False, axis=ax, other_func=other_func,
                          duplicate_axis=cls.use_multiple_axis)
            for ax in axes.flatten()[len(cls.list):]:
                ax.remove()
            fig.savefig(_repair_path(path))
            fig.clf()
        else:
            path = _path_for_multiple(path, "graphs")
            for i, rat in enumerate(tqdm(cls.list, desc="Saving Rationals")):
                pos = path.rfind(".")
                new_path = f"{path[:pos]}_{i}{path[pos:]}"
                rat.export_graph(new_path)

    @classmethod
    def export_evolution_graphs(cls, path="rationals_evolution.gif",
                                together=True, layout="auto", animated=True,
                                other_func=None):
        """
        Creates and saves an animated graph of the function evolution based \
        on the successive snapshots saved in `snapshot_list` for each \
        instanciated rational function.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rationals_evolution.gif"``
                together (bool):
                    If True, the graphs of every functions are stored in \
                    different files.\n
                    Default ``True``
                layout (tuple or 'auto'):
                    Grid layout of the figure. If "auto", one is generated.\
                    (see `layout`).\n
                    Default ``"auto"``
                animated (bool):
                    If True, creates an animated gif, else, different files \
                    are created.\n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other \
                    callable functions or a dictionary with the function \
                    name as key and the callable as value.\n
                    Default ``None``
        """
        if animated:
            if together:
                nb_sn = len(cls.list[0].snapshot_list)
                if any([len(rat.snapshot_list) != nb_sn for rat in cls.list]):
                    msg = "Seems that not all rationals have the same " \
                          "number of snapshots."
                    RationalWarning.warn(msg)
                import io
                from PIL import Image
                limits = []
                for i, rat in enumerate(cls.list):
                    if len(rat.snapshot_list) < 2:
                        msg = "Cannot save a gif as you have taken less " \
                              f"than 1 snapshot for rational n {i}"
                        print(msg)
                        return
                    limits.append(_get_frontiers(rat.snapshot_list,
                                                 other_func))
                if layout == "auto":
                    total = len(cls.list)
                    layout = _get_auto_axis_layout(total)
                if len(layout) != 2:
                    msg = 'layout should be either "auto" or a tuple of size 2'
                    raise TypeError(msg)
                fig = plt.gcf()
                gif_images = []
                seaborn_installed = True
                try:
                    import seaborn as sns
                except ImportError:
                    seaborn_installed = False
                    RationalImportSeabornWarning.warn()
                if seaborn_installed:
                    with sns.axes_style("whitegrid"):
                        figs = tuple(np.flip(np.array(layout)* (2, 3)))
                        fig, axes = plt.subplots(*layout, figsize=figs)
                else:
                    figs = tuple(np.flip(np.array(layout)* (2, 3)))
                    fig, axes = plt.subplots(*layout, figsize=figs)
                for ax in axes.flatten()[len(cls.list):]:
                    ax.remove()  # removes empty axes
                for i in range(nb_sn):
                    for rat, ax, lim in zip(cls.list, axes.flatten(), limits):
                        x_min, x_max, y_min, y_max = lim
                        input = np.arange(x_min, x_max, (x_max - x_min)/10000)
                        snap = rat.snapshot_list[i]
                        snap.show(x=input, other_func=other_func,
                                  display=False, axis=ax,
                                  duplicate_axis=cls.use_multiple_axis)
                        ax.set_xlim([x_min, x_max])
                        ax.set_ylim([y_min, y_max])
                    buf = io.BytesIO()
                    fig.set_tight_layout(True)
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    gif_images.append(Image.open(buf))
                    for i, ax in enumerate(fig.axes):
                        if i < len(cls.list):
                            ax.cla()
                        else:
                            ax.remove()
                if path[-4:] != ".gif":
                    path += ".gif"
                path = _repair_path(path)
                gif_images[0].save(path, save_all=True, duration=800, loop=0,
                                   append_images=gif_images[1:], optimize=False)
            else:
                path = _path_for_multiple(path, "graphs")
                bar_title = "Saving Rationals' evolutions"
                for i, rat in enumerate(tqdm(cls.list, desc=bar_title)):
                    pos = path.rfind(".")
                    if pos > 0:
                        new_path = f"{path[:pos]}_{i}{path[pos:]}"
                    else:
                        new_path = f"{path}_{i}"
                    rat.export_evolution_graph(new_path, True, other_func)
        else:  # not animated
            if path[-4:] == ".gif":
                path = path[-4:] + ".svg"
            path = _path_for_multiple(path, "evolution")
            if together:
                nb_sn = len(cls.list[0].snapshot_list)
                if any([len(rat.snapshot_list) != nb_sn for rat in cls.list]):
                    msg = "Seems that not all rationals have the " \
                          "same number of snapshots."
                    RationalWarning.warn(msg)
                for snap_number in range(nb_sn):
                    if "." in path:
                        ext = path.split(".")[-1]
                        main = ".".join(path.split(".")[:-1])
                        new_path = f"{main}_{snap_number}.{ext}"
                    else:
                        new_path = f"{path}_{snap_number}"
                    cls.export_graphs(new_path, together, layout, snap_number,
                                      other_func)
            else:
                for i, rat in enumerate(cls.list):
                    pos = path.rfind(".")
                    if pos > 0:
                        new_path = f"{path[pos:]}_{i}{path[:pos]}"
                    else:
                        new_path = f"{path}_{i}"
                    rat.export_evolution_graph(new_path, False, other_func)

    def export_evolution_graph(self, path="rational_evolution.gif",
                               animated=True, other_func=None):
        """
        Creates and saves an animated graph of the function evolution based \
        on the successive snapshots saved in `snapshot_list`.

        Arguments:
                path (str):
                    Complete path with name of the figure.\n
                    Default ``"rational_evolution.gif"``
                animated (bool):
                    Complete path with name of the figure.\n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other callable
                    functions or a dictionary with the function name as key
                    and the callable as value. \n
                    Default ``None``
        """
        if animated:
            import io
            from PIL import Image
            if len(self.snapshot_list) < 2:
                print("Cannot save a gif as you have taken less than 1 snapshot")
                return
            fig = plt.gcf()
            x_min, x_max, y_min, y_max = _get_frontiers(self.snapshot_list,
                                                        other_func)
            input = np.arange(x_min, x_max, (x_max - x_min)/10000)
            gif_images = []
            for i, snap in enumerate(self.snapshot_list):
                fig = snap.show(x=input, other_func=other_func, display=False,
                                duplicate_axis=self.use_multiple_axis)
                ax0 = fig.axes[0]
                ax0.set_xlim([x_min, x_max])
                ax0.set_ylim([y_min, y_max])
                buf = io.BytesIO()
                fig.set_tight_layout(True)
                plt.savefig(buf, format='png')
                buf.seek(0)
                gif_images.append(Image.open(buf))
                fig.clf()
            if path[-4:] != ".gif":
                path += ".gif"
            path = _repair_path(path)
            gif_images[0].save(path, save_all=True, duration=800, loop=0,
                               append_images=gif_images[1:], optimize=False)
        else:
            if path[-4:] == ".gif":
                path = path[-4:] + ".svg"
            path = _path_for_multiple(path, "evolution")
            for i, snap in enumerate(self.snapshot_list):
                pos = path.rfind(".")
                if pos > 0:
                    new_path = f"{path[pos:]}_{i}{path[:pos]}"
                else:
                    new_path = f"{path}_{i}"
                snap.save(path=new_path, other_func=other_func)

    def fit(self, function, x=None, show=False):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent \
        function of the provided one as close as possible to this rational \
        function.

        Arguments:
                function (callable):
                    The function you want to fit to rational.\n
                x (array):
                    The range on which the curves of the functions are fitted
                    together.\n
                    Default ``None``
                show (bool):
                    If  ``True``, plots the final fitted function and \
                    rational (using matplotlib).\n
                    Default ``False``
        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function \
                (vertical and horizontal scales and bias) \n
            dist: The final distance between the rational function and the \
            fitted one
        """
        if "rational.keras" in str(type(function)) or \
           "rational.torch" in str(type(function)):
            function = function.numpy()
        used_dist = False
        rational_numpy = self.numpy()
        if x is not None:
            (a, b, c, d), distance = rational_numpy.fit(function, x)
        else:
            if self.distribution is not None:
                freq, bins = _cleared_arrays(self.distribution)
                x = bins
                used_dist = True
            else:
                import numpy as np
                x = np.arange(-3., 3., 0.1)
            (a, b, c, d), distance = rational_numpy.fit(function, x)
        if show:
            def func(inp):
                return a * function(c * inp + d) + b

            if '__name__' in dir(function):
                func_label = function.__name__
            else:
                func_label = str(function)
            self.show(x, other_func={func_label: func})
        if self.best_fitted_function is None:
            self.best_fitted_function = function
            self.best_fitted_function_params = (a, b, c, d)
        return (a, b, c, d), distance

    def best_fit(self, functions_list, x=None, show=False):
        """
        Compute the distance between the rational and the functions in \
        `functions_list`, and return the one with the minimal the distance.

        Arguments:
                functions_list (list of callable):
                    The function you want to fit to rational.\n
                x (array):
                    The range on which the curves of the functions are fitted
                    together.\n
                    Default ``None``
                show (bool):
                    If  ``True``, plots the final fitted function and \
                    rational (using matplotlib).\n
                    Default ``False``
        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function \
                (vertical and horizontal scales and bias) \n
            dist: The final distance between the rational function and the \
            fitted one
        """
        if self.distribution is not None:
            freq, bins = _cleared_arrays(self.distribution)
            x = bins
        (a, b, c, d), distance = self.fit(functions_list[0], x=x, show=show)
        min_dist = distance
        print(f"{functions_list[0]}: {distance:>3}")
        params = (a, b, c, d)
        final_function = functions_list[0]
        for func in functions_list[1:]:
            (a, b, c, d), distance = self.fit(func, x=x, show=show)
            print(f"{func}: {distance:>3}")
            if min_dist > distance:
                min_dist = distance
                params = (a, b, c, d)
                final_func = func
                print(f"{func} is the new best fitted function")
        self.best_fitted_function = final_func
        self.best_fitted_function_params = params
        return final_func, (a, b, c, d)

    def numpy(self):
        """
        Returns a numpy version of this activation function.
        """
        raise NotImplementedError("the numpy method is not implemented for",
                                  " this class, only for the mother class")
