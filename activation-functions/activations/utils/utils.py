import numpy as np
from numpy import zeros, inf
import matplotlib.pyplot as plt
from .warnings import RationalWarning, RationalImportWarning, \
    RationalImportSeabornWarning, RationalImportScipyWarning


def _wrap_func(func, xdata, ydata, degrees, k=None):
    if k is not None:
        def func_wrapped(params):
            params1 = params[:degrees[0]-1]
            params2 = params[degrees[0]-1:]
            return func(xdata, k, params1, params2) - ydata
    else:
        def func_wrapped(params):
            params1 = params[:degrees[0]+1]
            params2 = params[degrees[0]+1:]
            return func(xdata, params1, params2) - ydata
    return func_wrapped


def _curve_fit(f, xdata, ydata, degrees, version, p0=None, absolute_sigma=False,
               method=None, jac=None, **kwargs):
    from scipy.optimize.optimize import OptimizeWarning
    from scipy.optimize._lsq.least_squares import prepare_bounds
    from scipy.optimize.minpack import leastsq, _wrap_jac
    bounds = (-np.inf, np.inf)
    lb, ub = prepare_bounds(bounds, np.sum(degrees))
    if p0 is None:
        if version == "C":
            p0 = np.ones(np.sum(degrees)+2)
        elif version == "RARE":
            degrees = (degrees[0] - 2, degrees[1])
            p0 = np.random.rand(np.sum(degrees)+1)
            # p0 = np.array([-0.0528, -1.6022, -1.0409, -1.3131, 0.3749, 0.7049, 0.3901, 0.6261])
            # p0 = np.array([-1., -1., -1., -1., 1., 1., 1., 1.])
        else:
            p0 = np.ones(np.sum(degrees)+1)
    method = 'lm'

    ydata = np.asarray_chkfinite(ydata, float)

    if isinstance(xdata, (list, tuple, np.ndarray)):
        # `xdata` is passed straight to the user-defined `f`, so allow
        # non-array_like `xdata`.
        xdata = np.asarray_chkfinite(xdata, float)

    k = kwargs.pop("k")
    func = _wrap_func(f, xdata, ydata, degrees, k)  # Modification here  !!!
    if callable(jac):
        jac = _wrap_jac(jac, xdata, None)
    elif jac is None and method != 'lm':
        jac = '2-point'

    if 'args' in kwargs:
        raise ValueError("'args' is not a supported keyword argument.")

    # Remove full_output from kwargs, otherwise we're passing it in twice.
    return_full = kwargs.pop('full_output', False)
    res = leastsq(func, p0, Dfun=jac, full_output=1, **kwargs)
    popt, pcov, infodict, errmsg, ier = res
    ysize = len(infodict['fvec'])
    cost = np.sum(infodict['fvec'] ** 2)
    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + errmsg)

    warn_cov = False
    if pcov is None:
        # indeterminate covariance
        pcov = zeros((len(popt), len(popt)), dtype=float)
        pcov.fill(inf)
        warn_cov = True
    elif not absolute_sigma:
        if ysize > p0.size:
            s_sq = cost / (ysize - p0.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(inf)
            warn_cov = True

    if warn_cov:
        RationalWarning.warn('Covariance of the parameters could not be estimated',
                             category=OptimizeWarning)

    if return_full:
        return popt, pcov, infodict, errmsg, ier
    else:
        return popt, pcov


def fit_rational_to_base_function(rational_func, ref_func, x, degrees=(5, 4), version="A"):
    y = ref_func(x)
    if version.lower() == "rare":
        k = np.abs(x)
    else:
        k = None
    final_params = _curve_fit(rational_func, x, y, degrees=degrees, version=version,
                              maxfev=10000000, k=k)[0]
    if k is not None:
        return np.array(final_params[:degrees[0]-1]), np.array(final_params[degrees[0]-1:])
    return np.array(final_params[:degrees[0]+1]), np.array(final_params[degrees[0]+1:])


def find_closest_equivalent(rational_func, new_func, x):
    """
    Compute the parameters a, b, c, and d that minimizes distance between the
    rational function and the other function on the range `x`

    Arguments:
            rational_func (callable):
                The rational function to consider.\n
            new_func (callable):
                The function you want to fit to rational.\n
            x (array):
                The range on which the curves of the functions are fitted
                together.\n
                Default ``True``
    Returns:
        tuple: ((a, b, c, d), dist) with: \n
        a, b, c, d: the parameters to adjust the function \
            (vertical and horizontal scales and bias) \n
        dist: The final distance between the rational function and the \
        fitted one
    """
    initials = np.array([1., 0., 1., 0.]) # a, b, c, d
    y = rational_func(x)
    from scipy.optimize import curve_fit
    import torch

    def equivalent_func(x_array, a, b, c, d):
        return a * new_func(c * x_array + d) + b
    params = curve_fit(equivalent_func, x, y, initials)
    a, b, c, d = params[0]
    final_func_output = np.array(equivalent_func(x, a, b, c, d))
    final_distance = np.sqrt(((y - final_func_output)**2).sum())
    return (a, b, c, d), final_distance



class Snapshot():
    """
    Snapshot to save, display, and export images of rational functions.
    Makes it easy to generate animations of the function through time, ... etc.

    Arguments:
            name (str):
                The name of Snapshot.
            rational (Rational):
                A rational function to save
            fitted_function (bool):
                If ``True``, displays the best fitted function if searched.
                Otherwise, returns it. \n
                Default ``True``
            other_func (callable):
                another function to be plotted or a list of other callable \
                functions or a dictionary with the function name as key \
                and the callable as value.
                Default ``None``
    Returns:
        Module: Rational module
    """

    _HIST_WARNED = False
    _SEABORN_WARNED = False
    _SCIPY_WARNED = False

    def __init__(self, name, rational, fitted_function=True, other_func=None):
        self.name = name
        self.rational = rational.numpy()
        self.use_kde = rational.use_kde
        self.range = None
        self.histogram = None
        self.rat_name = rational.func_name
        if rational.distribution is not None and \
           not rational.distribution.is_empty:
            from copy import deepcopy
            self.histogram = deepcopy(rational.distribution)
            msg = "Automatically clearing the distribution after snapshot"
            RationalWarning.warn(msg)
            rational.clear_hist()
        if fitted_function and rational.distribution is not None \
            and "best_fitted_function" in dir(rational):
            self.best_fitted_function = rational.best_fitted_function
            self.best_fitted_function_params = \
                rational.best_fitted_function_params
        else:
            self.best_fitted_function = None
            self.best_fitted_function_params = None
        self.other_func = other_func

    def show(self, x=None, fitted_function=True, other_func=None,
             display=True, tolerance=0.0001, title=None, axis=None,
             duplicate_axis=False, hist_color="#1f77b4"):
        """
        Show the function using `matplotlib`.

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                display (bool):
                    If ``True``, displays the graph.
                    Otherwise, returns a dictionary with functions informations. \n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other callable \
                    functions or a dictionary with the function name as key \
                    and the callable as value.
                    Default ``None``
                tolerance (float):
                    Tolerance the bins frequency.
                    If tolerance is 0.001, every frequency smaller than 0.001 \
                    will be cutted out of the histogram.\n
                    Default ``True``
                title (str)
                    If not `None`, title to be displayed on the figure.\n
                    Default ``None``
                axis (matplotlib.pyplot.axis):
                    axis to be plotted on. If None, creates one automatically.
                    Default ``None``
        """
        if x is not None:
            if "tensor" in str(x).lower():
                x = x.detach().cpu().numpy()
            elif x.dtype != float:
                x = x.astype(float)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
        elif x is None and self.range is not None:
            print("Snapshot: Using range from initialisation")
            x = self.range
        elif self.histogram is not None:
            x = np.array(self.histogram.bins, dtype=float)
            x = _cleared_arrays(self.histogram, tolerance)[1]
        elif x is None:
            x = np.arange(-3, 3, 0.01)
        y_rat = self.rational(x)
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            RationalImportSeabornWarning.warn()
        #  Rational
        if axis is None:
            ax = plt.gca()
        else:
            ax = axis
        # if duplicate_axis:
        #     oax = ax
        #     ax = ax.twinx()
        #     ax.set_zorder(oax.get_zorder()+1) # put ax in front of ax2
        #     ax.set_yticks([])
        #     ax.plot(x, y_rat, label=f"{self.rat_name}", zorder=2)
        ax.plot(x, y_rat, label=f"{self.rat_name}", zorder=2)
        if fitted_function and self.best_fitted_function is not None:
            if '__name__' in dir(self.best_fitted_function):
                func_label = self.best_fitted_function.__name__
            else:
                func_label = str(self.best_fitted_function)
            a, b, c, d = self.best_fitted_function_params
            y_bff = a * numpify(self.best_fitted_function, c * x + d) + b
            ax.plot(x, y_bff, "r-", label=f"Fitted {func_label}", zorder=2)
        #  Histogram
        if self.histogram is not None and hist_color is not None:
            weights = _cleared_arrays(self.histogram, tolerance)[1]
            ax2 = ax.twinx()
            ax2.set_yticks([])
            try:
                import scipy.stats as sts
                scipy_imported = True
            except ImportError:
                RationalImportScipyWarning.warn()
            if self.use_kde and scipy_imported:
                if len(x) > 5:
                    refined_bins = np.linspace(x[0], x[-1], 200)
                    kde_curv = self.histogram.kde()(refined_bins)
                    # ax2.plot(refined_bins, kde_curv, lw=0.1)
                    ax2.fill_between(refined_bins, kde_curv, alpha=0.15,
                                     color=hist_color)
                else:
                    print("The bin size is too big, bins contain too few "
                          "elements.\nbins:", x)
                    ax2.bar([], []) # in case of remove needed
            else:
                ax2.bar(x, weights/weights.max(), width=x[1] - x[0],
                        linewidth=0, alpha=0.3)
            ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
            ax.patch.set_visible(False)
        # Other funcs
        if other_func is None and self.other_func is not None:
            other_func = self.other_func
        if other_func is not None:
            if type(other_func) is dict:
                for func_label, func in other_func.items():
                    ax.plot(x, func(x), label=func)
            else:
                if type(other_func) is not list:
                    other_func = [other_func]
                for func in other_func:
                    if '__name__' in dir(func):
                        func_label = func.__name__
                    else:
                        func_label = str(func)
                    ax.plot(x, numpify(func, x), label=func_label)
            ax.legend(loc='upper right')
        if title is None:
            if not "snapshot" in self.name:
                ax.set_title(self.name)
        else:
            ax.set_title(f"{title}")
        if axis is None:
            if display:
                plt.show()
            else:
                return plt.gcf()

    def borders(self, x=None, fitted_function=True, other_func=None,
                tolerance=0.001):
        """
        Returns the borders x_min, x_max, y_min, y_max.

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other callable \
                    functions or a dictionary with the function name as key \
                    and the callable as value.
                    Default ``None``
                tolerance (float):
                    Tolerance the bins frequency.
                    If tolerance is 0.001, every frequency smaller than 0.001 \
                    will be cutted out of the histogram.\n
                    Default ``True``
        Returns:
            Module: Rational module
        """
        if x is not None:
            if x.dtype != float:
                x = x.astype(float)
            if not isinstance(x, np.ndarray):
                x = np.array(x)
        elif x is None and self.range is not None:
            x = self.range
        elif self.histogram is not None:
            x = np.array(self.histogram.bins, dtype=float)
            x = _cleared_arrays(self.histogram, tolerance)[1]
        elif x is None:
            x = np.arange(-3, 3, 0.01)
        y_rat = self.rational(x)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y_rat.min(), y_rat.max()
        if fitted_function and self.best_fitted_function is not None:
            a, b, c, d = self.best_fitted_function_params
            y_bff = a * numpify(self.best_fitted_function, c * x + d) + b
            y_min, y_max = min(y_min, y_bff.min()), max(y_max, y_bff.max())
        # Other funcs
        if other_func is None and self.other_func is not None:
            other_func = self.other_func
        if other_func is not None:
            if type(other_func) is dict:
                for func_label, func in other_func.items():
                    y_of = numpify(func, x)
                    y_min, y_max = min(y_min, y_of.min()), max(y_max, y_of.max())
            else:
                if type(other_func) is not list:
                    other_func = [other_func]
                for func in other_func:
                    y_of = numpify(func, x)
                    y_min, y_max = min(y_min, y_of.min()), max(y_max, y_of.max())
        return x_min, x_max, y_min, y_max


    def save(self, x=None, fitted_function=True, other_func=None,
             path=None, tolerance=0.001, title=None, format="svg"):
        """
        Saves an image of the snapshot.

        Arguments:
                x (range):
                    The range to print the function on.\n
                    Default ``None``
                fitted_function (bool):
                    If ``True``, displays the best fitted function if searched.
                    Otherwise, returns it. \n
                    Default ``True``
                other_func (callable):
                    another function to be plotted or a list of other callable \
                    functions or a dictionary with the function name as key \
                    and the callable as value.\n
                    Default ``None``
                tolerance (float):
                    Tolerance the bins frequency.
                    If tolerance is 0.001, every frequency smaller than 0.001 \
                    will be cutted out of the histogram.\n
                    Default ``True``
                title (str)
                    If not `None`, title to be displayed on the figure.\n
                    Default ``None``
                format (str)
                    The format of the figure, if not in the title.\n
                    Default ``svg``
        """
        fig = self.show(x, fitted_function, other_func, False, tolerance,
                        title)
        if path is None:
            path = self.name + f".{format}"
        elif "." not in path:
            path += f".{format}"
        path = _repair_path(path)
        fig.savefig(path)
        fig.clf()

    def __repr__(self):
        return f"Snapshot ({self.name})"


# def _cleared_arrays(hist, tolerance=0.001):
#     freq, bins = hist.normalize()
#     first = (freq > tolerance).argmax()
#     last = - (freq > tolerance)[::-1].argmax()
#     if last == 0:
#         return freq[first:], bins[first:]
#     return freq[first:last], bins[first:last]

def _cleared_arrays(weights, bins, tolerance=0.001):
    # weights, bins = hist.weights, hist.bins
    total = weights.sum()
    first = (weights > tolerance*total).argmax()
    last = - (weights > tolerance*total)[::-1].argmax()
    if last == 0:
        return weights[first:], bins[first:]
    return weights[first:last], bins[first:last]

def _repair_path(path):
    import os
    changed = False
    if os.path.exists(path):
        print(f'Path "{path}" exists')
        changed = True
    while os.path.exists(path):
        if "." in path:
            path_list = path.split(".")
            path_list[-2] = _increment_string(path_list[-2])
            path = '.'.join(path_list)
        else:
            path = _increment_string(path)
    if changed:
        print(f'Incremented, new path : "{path}"')
    if '/' in path:
        directory = "/".join(path.split("/")[:-1])
        if not os.path.exists(directory):
            print(f'Path "{directory}" does not exist, creating')
            os.makedirs(directory)
    return path


def _increment_string(string):
    if string[-1] in [str(i) for i in range(10)]:
        import re
        last_number = re.findall(r'\d+', string)[-1]
        return string[:-len(last_number)] + str(int(last_number) + 1)
    else:
        return string + "_2"


def _erase_suffix(string):
    if string[-1] in [str(i) for i in range(10)]:
        return "_".join(string.split("_")[:-1])
    else:
        return string


def _get_frontiers(snapshot_list, other_func=None, fitted_function=True,
                   tolerance=0.001):
    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    for snap in snapshot_list:
        x_mi, x_ma, y_mi, y_ma = snap.borders(fitted_function=fitted_function,
                                              other_func=other_func,
                                              tolerance=tolerance)
        if x_mi < x_min:
            x_min = x_mi
        if y_mi < y_min:
            y_min = y_mi
        if x_ma > x_max:
            x_max = x_ma
        if y_ma > y_max:
            y_max = y_ma
    span = y_max - y_min
    return x_min, x_max, y_min - 0.1 * span, y_max + 0.1 * span


def numpify(func, x):
    """
    Assert that the function is called and returns a numpy array
    """
    try:
        return np.array(func(x))
    except TypeError as tper:
        if "Tensor" in str(tper):
            import torch
            return func(torch.tensor(x)).detach().numpy()
        else:
            print("Doesn't know how to handle this type of data")
            raise tper


def _get_auto_axis_layout(nb_plots):
    if nb_plots == 1:
        return 1, 1
    mid = int(np.sqrt(nb_plots))
    for i in range(mid, 1, -1):
        mod = nb_plots % i
        if mod == 0:
            return i, nb_plots // i
    if mid * (mid + 1) >= nb_plots:
        return mid, mid + 1
    return mid + 1, mid + 1


def _path_for_multiple(path, suffix):
    from os import makedirs
    if "." in path:
        path_root = ".".join(path.split(".")[:-1])
        path_ext = "." + path.split(".")[-1]
    else:
        path_root = path
        path_ext = ""
    main_part = path_root.split("/")[-1]
    save_folder = _repair_path(f"{path_root}_{suffix}")
    makedirs(save_folder)
    return f"{save_folder}/{main_part}{path_ext}"


def _cupy_installed():
    try:
        import cupy as cp
        return True
    except ModuleNotFoundError:
        msg = "CuPy not found, please install it for fast input processing."
        url = "https://cupy.dev/"
        RationalImportWarning.warn(msg, url=url)
        return False
