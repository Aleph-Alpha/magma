import numpy as np


class Rational():
    """
    Rational activation function based on numpy

    Arguments:
            approx_func (str):
                The name of the approximated function for initialisation. \
                The different initialable functions are available in \
                `rational.rationals_config.json`. \n
                Default ``leaky_relu``.
            degrees (tuple of int):
                The degrees of the numerator (P) and denominator (Q).\n
                Default ``(5, 4)``
            version (str):
                Version of Rational to use. Rational(x) = P(x)/Q(x)\n
                `A`: Q(x) = 1 + \|b_1.x\| + \|b_2.x\| + ... + \|b_n.x\|\n
                `B`: Q(x) = 1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `C`: Q(x) = 0.1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `D`: like `B` with noise\n
                Default ``A``
    Returns:
        Module: Rational module
    """
    def __init__(self, approx_func="leaky_relu", degrees=(5, 4), version="A"):
        from rational.utils.get_weights import get_parameters
        w_numerator, w_denominator = get_parameters(version, degrees,
                                                    approx_func)
        self.numerator = w_numerator
        self.denominator = w_denominator
        self.init_approximation = approx_func
        self.degrees = degrees
        self.version = version

        if version == "A":
            rational_func = Rational_version_A
        elif version == "B":
            rational_func = Rational_version_B
        elif version == "C":
            rational_func = Rational_version_C
        elif version == "N":
            rational_func = Rational_version_N
        else:
            raise ValueError("version %s not implemented" % version)
        self.activation_function = rational_func

    def __call__(self, x):
        if type(x) is int:
            x = float(x)
        return self.activation_function(x, self.numerator, self.denominator)

    def torch(self, cuda=None, trainable=True, train_numerator=True,
              train_denominator=True):
        """
        Returns a torch version of this activation function.

        Arguments:
                cuda (bool):
                    Use GPU CUDA version. If None, use cuda if available on \
                    the machine\n
                    Default ``None``
                trainable (bool):
                    If the weights are trainable, i.e, if they are updated \
                    during backward pass\n
                    Default ``True``

        Returns:
            function: Rational torch function
        """
        from rational.torch import Rational as Rational_torch
        import torch.nn as nn
        import torch
        rtorch = Rational_torch(self.init_approximation, self.degrees,
                                cuda, self.version, trainable,
                                train_numerator, train_denominator)
        rtorch.numerator = nn.Parameter(torch.FloatTensor(self.numerator)
                                        .to(rtorch.device),
                                        requires_grad=trainable and train_numerator)
        rtorch.denominator = nn.Parameter(torch.FloatTensor(self.denominator)
                                          .to(rtorch.device),
                                          requires_grad=trainable and train_denominator)
        return rtorch

    def fit(self, function, x_range=np.arange(-3., 3., 0.1)):
        """
        Compute the parameters a, b, c, and d to have the neurally equivalent \
        function of the provided one as close as possible to this rational \
        function.

        Arguments:
                function (callable):
                    The function you want to fit to rational.
                x (array):
                    The range on which the curves of the functions are fitted \
                    together. \n
                    Default ``True``
                show (bool):
                    If  ``True``, plots the final fitted function and \
                    rational (using matplotlib) \n
                    Default ``False``

        Returns:
            tuple: ((a, b, c, d), dist) with: \n
            a, b, c, d: the parameters to adjust the function \
                (vertical and horizontal scales and bias) \n
            dist: The final distance between the rational function and the \
            fitted one.
        """
        from rational.utils import find_closest_equivalent
        (a, b, c, d), distance = find_closest_equivalent(self, function,
                                                         x_range)
        return (a, b, c, d), distance

    def __repr__(self):
        return (f"Rational Activation Function (Numpy version "
                f"{self.version}) of degrees {self.degrees}")

    def numpy(self):
        return self

    def show(self, input_range=None, display=True, distribution=None):
        """
        Show the function using `matplotlib`.

        Arguments:
                input_range (range):
                    The range to print the function on.\n
                    Default ``None``
                display (bool):
                    If ``True``, displays the graph.
                    Otherwise, returns it. \n
                    Default ``True``
        """
        import matplotlib.pyplot as plt
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError as e:
            print("seaborn not found on computer, install it for better",
                  "visualisation")
        ax = plt.gca()
        if input_range is None:
            if distribution is None:
                distribution = self.distribution
            if distribution is None:
                input_range = np.arange(-3, 3, 0.01)
            else:
                freq, bins = _cleared_arrays(distribution)
                if freq is None:
                    input_range = np.arange(-3, 3, 0.01)
                else:
                    ax2 = ax.twinx()
                    ax2.set_yticks([])
                    grey_color = (0.5, 0.5, 0.5, 0.6)

                    ax2.bar(bins, freq, width=bins[1] - bins[0],
                            color=grey_color, edgecolor=grey_color)
                    input_range = np.array(bins).float()
        else:
            input_range = np.array(input_range).float()
        outputs = self.activation_function(input_range, self.numerator,
                                           self.denominator, False)
        outputs_np = outputs.detach().cpu().numpy()
        ax.plot(input_range.detach().cpu().numpy(),
                outputs_np)
        if display:
            plt.show()
        else:
            return plt.gcf()

class EmbeddedRational():
    nb_rats = 2

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.successive_rats = []
        for i in range(self.nb_rats):
            rat = Rational(*args, **kwargs)
            self.successive_rats.append(rat)

    def __call__(self, x):
        if type(x) is int:
            x = float(x)
        for srat in self.successive_rats:
            x = srat.activation_function(x, srat.numerator, srat.denominator)
        return x

    def __repr__(self):
        return (f"EmbeddedRational Activation Function (Numpy version "
                f"{self.version}) of degrees {self.degrees}")


def Rational_version_A(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.ones_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += np.abs(d_array[i] * xi)
    return P/Q


def Rational_version_B(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += d_array[i] * xi
    Q = np.abs(Q) + np.ones_like(Q)
    return P/Q


def Rational_version_C(x, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        Q += d_array[i] * xi  # Here b0 is considered
        xi *= x
    Q = np.abs(Q) + np.full_like(Q, 0.1)
    return P/Q


def Rational_version_N(x, w_array, d_array):
    """
    Non safe version, original rational without norm
    """
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += d_array[i] * xi
    Q = Q + np.ones_like(Q)
    return P/Q


def RARE(x, k, w_array, d_array):
    xi = np.ones_like(x)
    P = np.ones_like(x) * w_array[0]
    for i in range(len(w_array) - 1):
        xi *= x
        P += w_array[i+1] * xi
    P *= (x-k)*(x+k)
    xi = np.ones_like(x)
    Q = np.zeros_like(x)
    for i in range(len(d_array)):
        xi *= x
        Q += d_array[i] * xi
    Q = np.abs(Q) + np.ones_like(Q)
    return P/Q
