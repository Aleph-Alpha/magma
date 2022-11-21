from typing import Type
from activations.torch import ActivationModule
import matplotlib.pyplot as plt
import torch
import numpy as np
from activations.torch.functions import _get_auto_axis_layout
from activations.utils.activation_logger import ActivationLogger


class GroupedActivations:
    def __init__(self, functions, name):
        self.checkType(functions)
        self.functions = functions
        self.logger = ActivationLogger(name)

    def _check_ActivationModule(self, object):
        if not isinstance(object, ActivationModule):
            raise TypeError(f"Activation Function {af} must be a ActivationModule Activation function")

    def add_func(self, af):
        self._check_ActivationModule(af)
        self.functions.append(af)

    def delete_func(self, objType):
        self._check_ActivationModule(objType)
        n = 0
        for func in self.functions: 
            if type(objType) is type(func):
                self.functions.remove(func)
                n += 1
        self.logger.info(f"Deleted {n} ActivationModule functions")

    def state_dicts(self, *args, **kwargs):
        state_dicts = []
        for func in self.functions:
            state_dicts.append(func.state_dict(*args, **kwargs))
        return state_dicts

    def show_all(self, x=None, fitted_function=True, other_func=None,
                 display=True, tolerance=0.001, title=None, axes=None,
                 layout="auto", writer=None, step=None, colors="#1f77b4"):
        functions = self.functions
        if axes is None:
            if layout == "auto":
                total = len(functions)
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
                self.logger.warn("Could not import seaborn")
                # RationalImportSeabornWarning.warn()
                fig, axes = plt.subplots(*layout, figsize=figs)
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            # if display:
            for ax in axes.flatten()[len(functions):]:
                ax.remove()
            axes = axes[:len(functions)]
        elif isinstance(axes, plt.Axes):
            axes = np.array([axes for _ in range(len(functions))])
            fig = plt.gcf()

        if isinstance(colors, str):
            colors = [colors]*len(axes.flatten())
        if isinstance(x, list):
            for rat, ax, x_rat, color in zip(functions, axes.flatten(), x, colors):
                rat.show(x_rat, fitted_function, other_func, False, tolerance,
                        title, axis=ax, writer=None, step=step,
                        color=color)
        else:
            for rat, ax, color in zip(functions, axes.flatten(), colors):
                rat.show(x, fitted_function, other_func, False, tolerance,
                        title, axis=ax, writer=None, step=step,
                        color=color)
        if title is not None:
            fig.suptitle(title, y=0.95)
        fig = plt.gcf()
        fig.tight_layout()
        # TODO: need to fix this
        """ if writer is not None:
            if step is None:
                step = cls._step
                cls._step += 1
            writer.add_figure(title, fig, step) """
        if display:
            # plt.legend()
            plt.show()
        else:
            return fig

    def save_all_inputs(self, *args, **kwargs):
        for func in self.functions:
            func.save_inputs(*args, **kwargs)

    def checkType(self, functions):
        if type(functions) is not list:
            raise TypeError(
                "Input functions must be a list of ActivationModule functions")
        for func in functions:
            if not isinstance(func, ActivationModule):
                raise TypeError(
                    f"Function {func} must be of type ActivationModule")

    def change_category(self, new_category):
        functions = self.functions
        all_functions = []
        if isinstance(functions, ActivationModule):
            functions.current_inp_category = new_category
            return
        elif type(functions) is list:
            all_functions = functions
        elif type(function) is dict:
            all_functions = _get_toplevel_functions(functions)
        else:
            raise TypeError("Type of function is not supported")
        for activationFunction in all_functions:
            activationFunction.current_inp_category = new_category


def _get_toplevel_functions(network):
    dict_afs = get_activations(network)
    functions = []
    all_keys = []
    for key in dict_afs:
        if "." not in key:
            all_keys.append(key)
    for top_key in all_keys:
        curr_obj = dict_afs[top_key]
        if type(curr_obj) is not list:
            functions.append(curr_obj)
        else:
            functions.extend(curr_obj)
    return functions



# TODO: there should be a way of isinstance(object, Container) instead,
#       but I couldn't find it.
def is_hierarchical(object):
    container_list = [torch.nn.modules.container.ModuleDict,
                      torch.nn.modules.container.ModuleList,
                      torch.nn.modules.container.Sequential,
                      torch.nn.modules.container.ParameterDict,
                      torch.nn.modules.container.ParameterList]

    for class_type in container_list:
        if isinstance(object, class_type):
            return True
    return False


def get_activations(network, top_level=False):
    """
    Retrieves a dictionary of all ActivationModule AFs present in the network

    Arguments: 
        network (torch.nn.Module):
            The network from which to retrieve all the ActivationModule AFs
    Returns:
        af_dict (dictionary):
            A dictionary containing as keys the names of the layer
            and as value a list of all AFs contained in the specific layer / object.\n 
            Duplicates will be in the dictionary, as hierarchical AFs are contained 
            in both the top-level hierarchy and lower-level hierarchies
    """
    found_instances = {}
    for name, object in network.named_children():
        if isinstance(object, ActivationModule):
            found_instances[name] = object
        elif is_hierarchical(object):
            found_instances[name] = _process_recursive(
                found_instances, name, object)
    return found_instances


def _process_recursive(original_dict, recName, recObject):
    af_list = []
    for name, object in recObject.named_children():
        if isinstance(object, ActivationModule):
            af_list.append(object)
        elif is_hierarchical(object):
            wholeName = recName + "." + name
            original_dict[wholeName] = _process_recursive(
                original_dict, name, object)
            af_list.extend(original_dict[wholeName])
    return af_list
