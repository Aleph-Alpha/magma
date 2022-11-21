import json
import os
from pathlib import Path
from .find_init_weights import find_weights
from .warnings import RationalImportError
import numpy as np
from termcolor import colored


known_functions = {
    "relu": lambda x: 0 if x < 0 else x,
    "leaky_relu": lambda x: x/100 if x < 0 else x,
    "lrelu": lambda x: x/100 if x < 0 else x,
    "normal": lambda x: 1/np.sqrt(2*np.pi) * np.exp(-.5*x**2),
}

def get_parameters(rational_version, degrees, approx_func, k=None):
    nd, dd = degrees
    if approx_func == "identity":
        return [0., 1.] + [0.] * (nd - 1), [0.] * dd
    elif approx_func == "ones":
        return [1.] * (nd + 1), [1.] * dd
    rational_full_name = f"Rational_version_{rational_version.upper()}{nd}/{dd}"
    if rational_version.lower() == "rare":
        nd -= 2
        rational_full_name += f"_k_{k}"
    config_file = '../rationals_config.json'
    config_file_dir = str(Path(os.path.abspath(__file__)).parent)
    url = "https://rational-activations.readthedocs.io/en/latest/tutorials/tutorials.1_find_weights_for_initialization.html"
    with open(os.path.join(config_file_dir, config_file)) as json_file:
        rationals_dict = json.load(json_file)
    if rational_full_name not in rationals_dict:
        if approx_func.lower() in known_functions:
            msg = f"Found {approx_func} but haven't computed its rational approximation yet for degrees {degrees}.\
            \nLet's do do it now:. \n--> More info:"
            print(colored(msg, "yellow"))
            print(colored(url, "blue"))
            find_weights(known_functions[approx_func.lower()], function_name=approx_func.lower(), degrees=degrees, version=rational_version)
            with open(os.path.join(config_file_dir, config_file)) as json_file:
                rationals_dict = json.load(json_file)
        else:
            msg = f"{rational_full_name} approximating \"{approx_func}\" not found in {config_file}.\
            \nWe need to add it.\nLet's do do it now. \n--> More info:"
            print(colored(msg, "yellow"))
            print(colored(url, "blue"))
            find_weights(known_functions[approx_func.lower()], function_name=approx_func.lower(), degrees=degrees, version=rational_version)
            with open(os.path.join(config_file_dir, config_file)) as json_file:
                rationals_dict = json.load(json_file)
    if approx_func not in rationals_dict[rational_full_name]:
        if approx_func.lower() in known_functions:
            msg = f"Found {approx_func} but haven't computed its rational approximation yet for degrees {degrees}.\
            \nLet's do do it now:. \n--> More info:"
            print(colored(msg, "yellow"))
            print(colored(url, "blue"))
            find_weights(known_functions[approx_func.lower()], function_name=approx_func.lower(), degrees=degrees, version=rational_version)
            with open(os.path.join(config_file_dir, config_file)) as json_file:
                rationals_dict = json.load(json_file)
        else:
            msg = f"{rational_full_name} approximating {approx_func} not found in {config_file}.\
            \nWe need to add it.\nLet's do do it now. \n--> More info:"
            print(colored(msg, "yellow"))
            print(colored(url, "blue"))
    params = rationals_dict[rational_full_name][approx_func]
    return params["init_w_numerator"], params["init_w_denominator"]


# P: [-0.01509308 -0.45776523 -0.29740746 -0.37517574 -0.07040943]
# Q: [0.37490264 0.70493863 0.39011436 0.62613085]
