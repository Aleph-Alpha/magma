"""
find_init_weights.py
====================================
Finding the weights of the to map an specific activation function
"""

import json
import numpy as np
from .utils import fit_rational_to_base_function
import torch
import os
from activations.numpy.rationals import Rational_version_A, Rational_version_B, \
    Rational_version_C, Rational_version_N, RARE


def plot_result(x_array, rational_array, target_array,
                original_func_name="Original function"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(x_array, rational_array, label="Rational approx")
    ax.plot(x_array, target_array, label=original_func_name)
    ax.legend()
    ax.grid()
    fig.show()


def append_to_config_file(params, approx_name, w_params, d_params, overwrite=None, k=None):
    rational_full_name = f'Rational_version_{params["version"]}{params["nd"]}/{params["dd"]}'
    if params["version"].lower() == 'rare':
        rational_full_name += f"_k_{k}"
    cfd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    with open(f'{cfd}/rationals_config.json') as json_file:
        rationals_dict = json.load(json_file)  # rational_version -> approx_func
    approx_name = approx_name.lower()
    if rational_full_name in rationals_dict:
        if approx_name in rationals_dict[rational_full_name]:
            if overwrite is None:
                overwrite = input(f'Rational_{params["version"]} approximation of {approx_name} already exist. \
                                  \nDo you want to replace it ? (y/n)') in ["y", "yes"]
            if not overwrite:
                print("Parameters not stored")
                return
        else:
            rationals_params = {"init_w_numerator": w_params.tolist(),
                                "init_w_denominator": d_params.tolist(),
                                "ub": params["ub"], "lb": params["lb"]}
            rationals_dict[rational_full_name][approx_name] = rationals_params
            with open(f'{cfd}/rationals_config.json', 'w') as outfile:
                json.dump(rationals_dict, outfile, indent=1)
            print("Parameters stored in rationals_config.json")
            return
    rationals_dict[rational_full_name] = {}
    rationals_params = {"init_w_numerator": w_params.tolist(),
                        "init_w_denominator": d_params.tolist(),
                        "ub": params["ub"], "lb": params["lb"]}
    rationals_dict[rational_full_name][approx_name] = rationals_params
    with open(f'{cfd}/rationals_config.json', 'w') as outfile:
        json.dump(rationals_dict, outfile, indent=1)
    print("Parameters stored in rationals_config.json")



def typed_input(text, type, choice_list=None):
    assert isinstance(text, str)
    while True:
        try:
            inp = input(text)
            typed_inp = type(inp)
            if choice_list is not None:
                assert typed_inp in choice_list
            break
        except ValueError:
            print(f"Please provide an type: {type}")
            continue
        except AssertionError:
            print(f"Please provide a value within {choice_list}")
            continue
    return typed_inp


FUNCTION = None


def find_weights(function, function_name=None, degrees=None, bounds=None,
                 version=None, plot=None, save=None, overwrite=None, k=None):
    """
    Finds the weights of the numerator and the denominator of the rational function.
    Beside `function`, all parameters can be left to the default ``None``. \n
    In this case, user is asked to provide the params interactively.

    Arguments:
            function (callable):
                The function to approximate (e.g. from torch.functional).\n
            function_name (str):
                The name of this function (used at Rational initialisation)\n
            degrees (tuple of int):
                The degrees of the numerator (P) and denominator (Q).\n
                Default ``None``
            bounds (tuple of int):
                The bounds to approximate on (e.g. (-3,3)).\n
                Default ``None``
            version (str):
                Version of Rational to use. Rational(x) = P(x)/Q(x)\n
                `A`: Q(x) = 1 + \|b_1.x\| + \|b_2.x\| + ... + \|b_n.x\|\n
                `B`: Q(x) = 1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `C`: Q(x) = 0.1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `D`: like `B` with noise\n
            plot (bool):
                If True, plots the fitted and target functions.
                Default ``None``
            save (bool):
                If True, saves the weights in the config file.
                Default ``None``
            save (bool):
                If True, if weights already exist for this configuration, they are overwritten.
                Default ``None``
    Returns:
        tuple: (numerator, denominator) if not `save`, otherwise `None` \n

    """
    # To be changed by the function you want to approximate
    if function_name is None:
        function_name = input("approximated function name: ")
    FUNCTION = function

    def function_to_approx(x):
        # return np.heaviside(x, 0)
        x = torch.tensor(x)
        return FUNCTION(x)

    if degrees is None:
        nd = typed_input("degree of the numerator P: ", int)
        dd = typed_input("degree of the denominator Q: ", int)
        degrees = (nd, dd)
    else:
        nd, dd = degrees
    if bounds is None:
        print("On what range should the function be approximated ?")
        lb = typed_input("lower bound: ", float)
        ub = typed_input("upper bound: ", float)

    else:
        lb, ub = bounds
    nb_points = 100000
    step = (ub - lb) / nb_points
    x = np.arange(lb, ub, step)
    if version is None:
        version = typed_input("Rational Version: ", str,
                              ["A", "B", "C", "D", "N", "RARE"])
    if version == 'A':
        rational = Rational_version_A
    elif version == 'B':
        rational = Rational_version_B
    elif version == 'C':
        rational = Rational_version_C
    elif version == 'D':
        rational = Rational_version_B
    elif version == 'N':
        rational = Rational_version_N
    elif version == 'RARE':
        rational = RARE
        if k is None:
            print("What is the k limit of the RARE function ?")
            k = typed_input("k: ", float)

    w_params, d_params = fit_rational_to_base_function(rational, function_to_approx, x,
                                                       degrees=degrees, version=version)
    print(f"Found coeffient :\nP: {w_params}\nQ: {d_params}")
    if plot is None:
        plot = input("Do you want a plot of the result (y/n)") in ["y", "yes"]
    if plot:
        if version == 'RARE':
            plot_result(x, rational(x, k, w_params, d_params), function_to_approx(x),
                        function_name)
        else:
            plot_result(x, rational(x, w_params, d_params), function_to_approx(x),
                        function_name)
    params = {"version": version, "name": function_name, "ub": ub, "lb": lb,
              "nd": nd, "dd": dd}
    if save is None:
        save = input("Do you want to store them in the json file ? (y/n)") in ["y", "yes"]
    if save:
        append_to_config_file(params, function_name, w_params, d_params, overwrite, k)
    else:
        print("Parameters not stored")
        return w_params, d_params
