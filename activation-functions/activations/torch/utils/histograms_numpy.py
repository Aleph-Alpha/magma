import numpy as np
import scipy.stats as sts


class Histogram():
    """
    Input Histograms, used to retrieve the input of Rational Activations
    """
    def __init__(self, bin_size="auto", random_select=False):
        self.bins = np.array([])
        self.weights = np.array([], dtype=np.uint32)
        self._is_empty = True
        self._verbose = False
        if bin_size == "auto":
            self._auto_bin_size = True
            self._bin_size = 0.0001
            self._rd = 4
        else:
            self._auto_bin_size = False
            self._bin_size = float(bin_size)
            self._rd = int(np.log10(1./bin_size).item())
            self._fill_iplm = self._first_time_fill

    def fill_n(self, input):
        self._fill_iplm(input.detach().numpy())

    def _first_time_fill(self, new_input):
        range_ext = np.around(new_input.min() - self._bin_size / 2, self._rd), \
                    np.around(new_input.max() + self._bin_size / 2, self._rd)
        bins_array = np.arange(range_ext[0], range_ext[1] + self._bin_size,
                               self._bin_size)
        weights, bins = np.histogram(new_input, bins_array)
        if self._auto_bin_size:
            self._rd = int(np.log10(1./(range_ext[1] - range_ext[0])).item()) + 2
            self._bin_size = 1./(10**self._rd)
            range_ext = np.around(new_input.min() - self._bin_size / 2, self._rd), \
                        np.around(new_input.max() + self._bin_size / 2, self._rd)
            bins_array = np.arange(range_ext[0], range_ext[1] + self._bin_size,
                                   self._bin_size)
            weights, bins = np.histogram(new_input, bins_array)
        self.weights, self.bins = weights, bins[:-1]
        self._is_empty = False
        self._fill_iplm = self._update_hist

    def _update_hist(self, new_input):
        range_ext = np.around(new_input.min() - self._bin_size / 2, self._rd), \
                    np.around(new_input.max() + self._bin_size / 2, self._rd)
        bins_array = np.arange(range_ext[0], range_ext[1] + self._bin_size,
                               self._bin_size)
        weights, bins = np.histogram(new_input, bins_array)
        self.weights, self.bins = concat_hists(self.weights, self.bins,
                                               weights, bins[:-1],
                                               self._bin_size, self._rd)

    def __repr__(self):
        if self.is_empty:
            rtrn = "Empty Histogram"
        else:
            rtrn = f"Histogram on range {self.bins[0]}, {self.bins[-1]}, of " + \
                   f"bin_size {self._bin_size}, with {self.weights.sum()}" + \
                   f"elements"
        if self._verbose:
            rtrn += f" {hex(id(self))}"
        return rtrn

    @property
    def total(self):
        return self.weights.sum()

    @property
    def is_empty(self):
        if self._is_empty is True and len(self.bins) > 0:
            self._is_empty = False
        return self._is_empty

    def normalize(self, numpy=True, nb_output=100):
        """

        """
        if nb_output is not None and nb_output < len(self.bins):
            div = len(self.bins) // nb_output
            if len(self.bins) % div == 0:
                weights = np.nanmean(self.weights.reshape(-1, div), axis=1)
                last = self.bins[-1]
            else:
                to_add = div - self.weights.size % div
                padded = np.pad(self.weights, (0, to_add), mode='constant',
                                constant_values=np.NaN).reshape(-1, div)
                weights = np.nanmean(padded, axis=1)
                last = self.bins[-1] + self._bin_size * to_add
            bins = np.linspace(self.bins[0], last, len(weights),
                               endpoint=False)
            return weights / weights.sum(), bins
        else:
            return self.weights / self.weights.sum(), self.bins

    def _from_physt(self, phystogram):
        if (phystogram.bin_sizes == phystogram.bin_sizes[0]).all():
            self._bin_size = phystogram.bin_sizes[0]
        self.bins = np.array(phystogram.bin_left_edges)
        self.weights = np.array(phystogram.frequencies)
        return self

    def kde(self):
        kde = sts.gaussian_kde(self.bins, bw_method=0.13797296614612148,
                               weights=self.weights)
        return kde.pdf

    @property
    def bin_size(self):
        return self._bin_size

    @bin_size.setter
    def bin_size(self, value):
        self._bin_size = value
    #     if value < 0:
    #         raise TypeError("Cannot have a negative bin size")
    #     elif value < self._bin_size:
    #         raise Error("Cannot decrease the bin size, only increase it")
    #     self._bin_size = value
    #     bin_min = min(self.bins) // value * value
    #     import ipdb; ipdb.set_trace()



class NeuronsHistogram():
    """
    Input Histograms, used to retrieve the input of Rational Activations
    """
    def __init__(self, bin_size="auto", random_select=False, nb_neurons="auto"):
        self._is_empty = True
        self._verbose = False
        self.nb_neurons = nb_neurons
        if nb_neurons != "auto":
            self.__bins = [np.array([]) for _ in range(nb_neurons)]
            self.__weights = [np.array([], dtype=np.uint32) for _ in range(nb_neurons)]
        if bin_size == "auto":
            self._auto_bin_size = True
            self._bin_size = 0.0001
            self._rd = 4
        else:
            self._auto_bin_size = False
            self._bin_size = bin_size
            self._rd = int(np.log10(1./bin_size).item())
        self._fill_iplm = self._first_time_fill

    def fill_n(self, input):
        self._fill_iplm(input.T.detach().numpy())

    def _first_time_fill(self, new_input):
        n_neurs = new_input.shape[0]
        if n_neurs != self.nb_neurons:
            if self.nb_neurons != "auto":
                msg = f"It seems that the layer currently has {n_neurs} neurons.\n"
                msg += "Automatically changing."
                print(colored(msg, "yellow"))
            self.nb_neurons = n_neurs
            self.__bins = [np.array([]) for _ in range(n_neurs)]
            self.__weights = [np.array([], dtype=np.uint32) for _ in range(n_neurs)]
        if self._auto_bin_size:
            # on the complete input to get the total range
            range_ext = np.around(new_input.min() - self._bin_size / 2, self._rd), \
                        np.around(new_input.max() + self._bin_size / 2, self._rd)
            self._rd = int(np.log10(1./(range_ext[1] - range_ext[0])).item()) + 2
            self._bin_size = 1./(10**self._rd)
            range_ext = np.around(new_input.min() - self._bin_size / 2, self._rd), \
                        np.around(new_input.max() + self._bin_size / 2, self._rd)
            bins_array = np.arange(range_ext[0], range_ext[1] + self._bin_size,
                                   self._bin_size)
        for n, neur_inp in enumerate(new_input):
            range_ext = np.around(neur_inp.min() - self._bin_size / 2, self._rd), \
                        np.around(neur_inp.max() + self._bin_size / 2, self._rd)
            bins_array = np.arange(range_ext[0], range_ext[1] + self._bin_size,
                                   self._bin_size)
            weights, bins = np.histogram(neur_inp, bins_array)
            self.__weights[n], self.__bins[n] = weights, bins[:-1]
        self._is_empty = False
        self._fill_iplm = self._update_hist

    def _update_hist(self, new_input):
        for n, (neur_inp, neur_b, neur_w) in enumerate(zip(new_input, self.__bins, self.__weights)):
            range_ext = np.around(new_input.min() - self._bin_size / 2, self._rd), \
                        np.around(new_input.max() + self._bin_size / 2, self._rd)
            bins_array = np.arange(range_ext[0], range_ext[1] + self._bin_size,
                                   self._bin_size)
            weights, bins = np.histogram(new_input, bins_array)
            self.__weights[n], self.__bins[n] = concat_hists(neur_w, neur_b,
                                                             weights, bins[:-1],
                                                             self._bin_size, self._rd)

    def __repr__(self):
        if self.is_empty:
            rtrn = "Empty Layer Histogram"
        else:
            rtrn = f"Layer histogram on {self.nb_neurons} neurons"
        if self._verbose:
            rtrn += f" {hex(id(self))}"
        return rtrn

    @property
    def range(self):
        x_min = float(min([b[0] for b in self.__bins]))
        x_max = float(max([b[-1] for b in self.__bins]))
        return np.arange(x_min, x_max, self._bin_size/100)

    @property
    def bins(self):
        return [b.flatten() for b in self.__bins]

    # @bins.setter
    # def bins(self, var):
    #     if isinstance(var, np.ndarray):
    #         self.__bins = cp.array(var)
    #     else:
    #         self.__bins = var

    @property
    def weights(self):
        return [w.flatten() for w in self.__weights]

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
            return (self.__weights / self.__weights.sum()).flatten(), \
                   self.__bins
        else:
            return self.__weights / self.__weights.sum(), self.__bins

    def kde(self, n):
        kde = sts.gaussian_kde(self.__bins[n], bw_method=0.13797296614612148,
                               weights=self.__weights[n])
        return kde.pdf


def concat_hists(weights1, bins1, weights2, bins2, bin_size, rd):
    min1, max1 = np.around(bins1[0], rd), np.around(bins1[-1], rd)
    min2, max2 = np.around(bins2[0], rd), np.around(bins2[-1], rd)
    mini, maxi = min(min1, min2), max(max1, max2)
    new_bins = np.arange(mini, maxi + bin_size*0.9, bin_size)  # * 0.9 to avoid unexpected random inclusion of last element
    if min1 - mini != 0 and maxi - max1 != 0:
        ext1 = np.pad(weights1, (np.int(np.around((min1 - mini) / bin_size)),
                                 np.int(np.around((maxi - max1) / bin_size))),
                      'constant', constant_values=0)
    elif min1 - mini != 0:
        ext1 = np.pad(weights1, (np.int(np.around((min1 - mini) / bin_size)),
                                 0), 'constant', constant_values=0)
    elif maxi - max1 != 0:
        ext1 = np.pad(weights1, (0,
                                 np.int(np.around((maxi - max1) / bin_size))),
                      'constant', constant_values=0)
    else:
        ext1 = weights1
    if min2 - mini != 0 and maxi - max2 != 0:
        ext2 = np.pad(weights2, (np.int(np.around((min2 - mini) / bin_size)),
                                 np.int(np.around((maxi - max2) / bin_size))),
                      'constant', constant_values=0)
    elif min2 - mini != 0:
        ext2 = np.pad(weights2, (np.int(np.around((min2 - mini) / bin_size)),
                                 0), 'constant', constant_values=0)
    elif maxi - max2 != 0:
        ext2 = np.pad(weights2, (0,
                                 np.int(np.around((maxi - max2) / bin_size))),
                      'constant', constant_values=0)
    else:
        ext2 = weights2
    new_ext = ext1 + ext2
    return new_ext, new_bins
