import warnings
from inspect import stack
try:
    from termcolor import colored
    termc_installed = True
except ImportError:
    warnings.showwarning("Install termcolor for better rational warning " \
                         "colors", ImportWarning,
                         "warnings.py", "7")
    termc_installed = False
    colored = lambda msg, col: msg


class RationalWarning(UserWarning):
    base_category = UserWarning
    done_list = []

    def __init__(self):
        pass

    @classmethod
    def warn(cls, message=None, category=None, url=None):
        if message not in cls.done_list:
            if message is None:
                message = cls.message
            if url is not None:
                url = colored(url, "blue")
                message += f"\n -> More info: {url}"
            cls.done_list.append(message)
            if termc_installed:
                message = colored(message, "yellow")
            _cf = stack()
            if category is None:
                category = cls.base_category
            warnings.showwarning(message, category, _cf[1][1], _cf[1][2])


class RationalImportWarning(RationalWarning):
    base_category = ImportWarning

class RationalLoadWarning(RationalWarning):
    base_category = ImportWarning

class RationalImportSeabornWarning(RationalImportWarning):
    message = "Could not find Seaborn installed, please install it for " \
              "better visualisation"


class RationalImportScipyWarning(RationalImportWarning):
    message = "Could not find Scipy installed, please install it for " \
              "PDE computation."


class RationalImportError(ImportError):
    def __init__(self, message, url=None):
        if termc_installed:
            message = colored(message, "red")
            url = colored(url, "blue")
        if url is not None:
            message += f"\n -> More info: {url}"
        super().__init__(message)
