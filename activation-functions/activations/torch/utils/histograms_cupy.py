import cupy as cp
from torch.utils.dlpack import to_dlpack
import scipy.stats as sts
import numpy as np
from termcolor import colored


class Histogram():
    """
    Input Histograms, used to retrieve the input of Rational Activations
    """

    def __init__(self, bin_size="auto", random_select=False):
        self.bins = cp.array([])
        self.weights = cp.array([], dtype=cp.uint32)
        self._is_empty = True
        self._verbose = False
        if bin_size == "auto":
            self._auto_bin_size = True
            self.bin_size = 0.0001
            self._rd = 4
            self._fill_iplm = self._first_time_fill
        else:
            self._auto_bin_size = False
            self.bin_size = float(bin_size)
            rd = cp.log10(1./bin_size).item()
            if abs(round(rd)-rd) > 0.0001:
                print("The bin size of the histogram should be 0.01, 0.1, 1., 10., ... etc")
                exit()
            self._rd = round(rd)
            self._fill_iplm = self._first_time_fill

    def fill_n(self, input):
        self._fill_iplm(cp.fromDlpack(to_dlpack(input)))

    def _first_time_fill(self, new_input):
        range_ext = cp.around(new_input.min() - self.bin_size / 2, self._rd), \
                    cp.around(new_input.max() + self.bin_size / 2, self._rd)
        bins_array = cp.arange(range_ext[0], range_ext[1] + self.bin_size,
                               self.bin_size)
        weights, bins = cp.histogram(new_input, bins_array)
        if self._auto_bin_size:
            self._rd = int(cp.log10(1./(range_ext[1] - range_ext[0])).item()) + 2
            self.bin_size = 1./(10**self._rd)
            range_ext = cp.around(new_input.min() - self.bin_size / 2, self._rd), \
                        cp.around(new_input.max() + self.bin_size / 2, self._rd)
            bins_array = cp.arange(range_ext[0], range_ext[1] + self.bin_size,
                                   self.bin_size)
            weights, bins = cp.histogram(new_input, bins_array)
        self.weights, self.bins = weights, bins[:-1]
        self._is_empty = False
        self._fill_iplm = self._update_hist

    def _update_hist(self, new_input):
        range_ext = cp.around(new_input.min() - self.bin_size / 2, self._rd), \
                    cp.around(new_input.max() + self.bin_size / 2, self._rd)
        bins_array = cp.arange(range_ext[0], range_ext[1] + self.bin_size,
                               self.bin_size)
        weights, bins = cp.histogram(new_input, bins_array)
        self.weights, self.bins = concat_hists(self.__weights, self.__bins,
                                               weights, bins[:-1],
                                               self.bin_size, self._rd)

    def __repr__(self):
        if self.is_empty:
            rtrn = "Empty Histogram"
        else:
            rtrn = f"Histogram on range {self.bins[0]}, {self.bins[-1]}, of " + \
                   f"bin_size {self.bin_size}, with {self.weights.sum()}" + \
                   f"elements"
        if self._verbose:
            rtrn += f" {hex(id(self))}"
        return rtrn

    def empty(self):
        """
        empties the histogram
        """
        self._is_empty = True
        self.bins = cp.array([])
        self.weights = cp.array([], dtype=cp.uint32)
        self._fill_iplm = self._first_time_fill

    @property
    def bins(self):
        return self.__bins.get().flatten()

    @bins.setter
    def bins(self, var):
        if isinstance(var, np.ndarray):
            self.__bins = cp.array(var)
        else:
            self.__bins = var

    @property
    def is_empty(self):
        if self._is_empty is True and len(self.bins) > 0:
            self._is_empty = False
        return self._is_empty

    @property
    def total(self):
        return self.weights.sum()

    @property
    def weights(self):
        return self.__weights.get().flatten()

    @weights.setter
    def weights(self, var):
        if isinstance(var, np.ndarray):
            self.__weights = cp.array(var)
        else:
            self.__weights = var

    def normalize(self, numpy=True):
        if numpy:
            return (self.__weights / self.__weights.sum()).get().flatten(), \
                   self.bins
        else:
            return self.__weights / self.__weights.sum(), self.__bins

    def kde(self):
        kde = sts.gaussian_kde(self.bins, bw_method=0.13797296614612148,
                               weights=self.weights)
        return kde.pdf

    @property
    def range(self):
        x_min = float(self.__bins[0])
        x_max = float(self.__bins[-1])
        return np.arange(x_min, x_max, self.bin_size/100)


class NeuronsHistogram():
    """
    Input Histograms, used to retrieve the input of Rational Activations
    """
    def __init__(self, bin_size="auto", random_select=False, nb_neurons="auto"):
        self._is_empty = True
        self._verbose = False
        self.nb_neurons = nb_neurons
        if nb_neurons != "auto":
            self.__bins = [cp.array([]) for _ in range(nb_neurons)]
            self.__weights = [cp.array([], dtype=cp.uint32) for _ in range(nb_neurons)]
        if bin_size == "auto":
            self._auto_bin_size = True
            self.bin_size = 0.0001
            self._rd = 4
        else:
            self._auto_bin_size = False
            self.bin_size = bin_size
            self._rd = int(cp.log10(1./bin_size).item())
        self._fill_iplm = self._first_time_fill

    def fill_n(self, input):
        self._fill_iplm(cp.fromDlpack(to_dlpack(input.T)))

    def _first_time_fill(self, new_input):
        n_neurs = new_input.shape[0]
        if n_neurs != self.nb_neurons:
            if self.nb_neurons != "auto":
                msg = f"It seems that the layer currently has {n_neurs} neurons.\n"
                msg += "Automatically changing."
                print(colored(msg, "yellow"))
            self.nb_neurons = n_neurs
            self.__bins = [cp.array([]) for _ in range(n_neurs)]
            self.__weights = [cp.array([], dtype=cp.uint32) for _ in range(n_neurs)]
        if self._auto_bin_size:
            # on the complete input to get the total range
            range_ext = cp.around(new_input.min() - self.bin_size / 2, self._rd), \
                        cp.around(new_input.max() + self.bin_size / 2, self._rd)
            self._rd = int(cp.log10(1./(range_ext[1] - range_ext[0])).item()) + 2
            self.bin_size = 1./(10**self._rd)
            range_ext = cp.around(new_input.min() - self.bin_size / 2, self._rd), \
                        cp.around(new_input.max() + self.bin_size / 2, self._rd)
            bins_array = cp.arange(range_ext[0], range_ext[1] + self.bin_size,
                                   self.bin_size)
        for n, neur_inp in enumerate(new_input):
            range_ext = cp.around(neur_inp.min() - self.bin_size / 2, self._rd), \
                        cp.around(neur_inp.max() + self.bin_size / 2, self._rd)
            bins_array = cp.arange(range_ext[0], range_ext[1] + self.bin_size,
                                   self.bin_size)
            weights, bins = cp.histogram(neur_inp, bins_array)
            self.__weights[n], self.__bins[n] = weights, bins[:-1]
        self._is_empty = False
        self._fill_iplm = self._update_hist

    def _update_hist(self, new_input):
        for n, (neur_inp, neur_b, neur_w) in enumerate(zip(new_input, self.__bins, self.__weights)):
            range_ext = cp.around(new_input.min() - self.bin_size / 2, self._rd), \
                        cp.around(new_input.max() + self.bin_size / 2, self._rd)
            bins_array = cp.arange(range_ext[0], range_ext[1] + self.bin_size,
                                   self.bin_size)
            weights, bins = cp.histogram(new_input, bins_array)
            self.__weights[n], self.__bins[n] = concat_hists(neur_w, neur_b,
                                                             weights, bins[:-1],
                                                             self.bin_size, self._rd)

    def __repr__(self):
        if self.is_empty:
            rtrn = "Empty Layer Histogram"
        else:
            # rtrn = f"Layer Histogram on range {self.bins[0]}, {self.bins[-1]}, of " + \
            #        f"bin_size {self.bin_size}, with {self.weights.sum()} total" + \
            #        f"elements"
            rtrn = f"Layer histogram on {self.nb_neurons} neurons"
        if self._verbose:
            rtrn += f" {hex(id(self))}"
        return rtrn

    @property
    def range(self):
        x_min = float(min([b[0] for b in self.__bins]))
        x_max = float(max([b[-1] for b in self.__bins]))
        return np.arange(x_min, x_max, self.bin_size/100)

    @property
    def bins(self):
        return [b.get().flatten() for b in self.__bins]

    def empty(self):
        """
        empties the histogram
        """
        self._fill_iplm = self._first_time_fill
        self._is_empty = True

    # @bins.setter
    # def bins(self, var):
    #     if isinstance(var, np.ndarray):
    #         self.__bins = cp.array(var)
    #     else:
    #         self.__bins = var

    @property
    def weights(self):
        return [w.get().flatten() for w in self.__weights]

    @property
    def is_empty(self):
        if self._is_empty is True and len(self.__bins[0]) > 0:
            self._is_empty = False
        return self._is_empty

    @property
    def total(self):
        return self.__weights.sum()

    # @weights.setter
    # def weights(self, var):
    #     if isinstance(var, np.ndarray):
    #         self.__weights = cp.array(var)
    #     else:
    #         self.__weights = var

    def normalize(self, numpy=True):
        if numpy:
            return (self.__weights / self.__weights.sum()).get().flatten(), \
                   self.__bins
        else:
            return self.__weights / self.__weights.sum(), self.__bins

    def kde(self, n):
        kde = sts.gaussian_kde(self.__bins[n].get(), bw_method=0.13797296614612148,
                               weights=self.__weights[n].get())
        return kde.pdf


def concat_hists(weights1, bins1, weights2, bins2, bin_size, rd):
    min1, max1 = cp.around(bins1[0], rd), cp.around(bins1[-1], rd)
    min2, max2 = cp.around(bins2[0], rd), cp.around(bins2[-1], rd)
    mini, maxi = min(min1, min2), max(max1, max2)
    new_bins = cp.arange(mini, maxi + bin_size*0.9, bin_size)  # * 0.9 to avoid unexpected random inclusion of last element
    if min1 - mini != 0 and maxi - max1 != 0:
        ext1 = cp.pad(weights1, (cp.int(cp.around((min1 - mini) / bin_size)),
                                 cp.int(cp.around((maxi - max1) / bin_size))),
                      'constant', constant_values=0)
    elif min1 - mini != 0:
        ext1 = cp.pad(weights1, (cp.int(cp.around((min1 - mini) / bin_size)),
                                 0), 'constant', constant_values=0)
    elif maxi - max1 != 0:
        ext1 = cp.pad(weights1, (0,
                                 cp.int(cp.around((maxi - max1) / bin_size))),
                      'constant', constant_values=0)
    else:
        ext1 = weights1
    if min2 - mini != 0 and maxi - max2 != 0:
        ext2 = cp.pad(weights2, (cp.int(cp.around((min2 - mini) / bin_size)),
                                 cp.int(cp.around((maxi - max2) / bin_size))),
                      'constant', constant_values=0)
    elif min2 - mini != 0:
        ext2 = cp.pad(weights2, (cp.int(cp.around((min2 - mini) / bin_size)),
                                 0), 'constant', constant_values=0)
    elif maxi - max2 != 0:
        ext2 = cp.pad(weights2, (0,
                                 cp.int(cp.around((maxi - max2) / bin_size))),
                      'constant', constant_values=0)
    else:
        ext2 = weights2
    new_ext = ext1 + ext2
    return new_ext, new_bins
