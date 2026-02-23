from sklearn.decomposition import PCA
from scipy.signal import butter, lfilter, filtfilt
import copy
import pandas as pd

class LowPassFilter:
    def low_pass_filter(
        self,
        data_table,
        col,
        sampling_frequency,
        cutoff_frequency,
        order=5,
        phase_shift=True,
    ):
        nyq = 0.5 * sampling_frequency
        cut = cutoff_frequency / nyq

        b, a = butter(order, cut, btype="low", output="ba", analog=False)
        if phase_shift:
            data_table[col + "_lowpass"] = filtfilt(b, a, data_table[col])
        else:
            data_table[col + "_lowpass"] = lfilter(b, a, data_table[col])
        return data_table


class PrincipalComponentAnalysis:

    pca = []

    def __init__(self):
        self.pca = []

    def normalize_dataset(self, data_table, columns):
        dt_norm = copy.deepcopy(data_table)
        for col in columns:
            dt_norm[col] = (data_table[col] - data_table[col].mean()) / (
                data_table[col].max()
                - data_table[col].min()
                # data_table[col].std()
            )
        return dt_norm

    def determine_pc_explained_variance(self, data_table, cols):

        dt_norm = self.normalize_dataset(data_table, cols)

        self.pca = PCA(n_components=len(cols))
        self.pca.fit(dt_norm[cols])
        return self.pca.explained_variance_ratio_

    def apply_pca(self, data_table, cols, number_comp):

        dt_norm = self.normalize_dataset(data_table, cols)

        self.pca = PCA(n_components=number_comp)
        self.pca.fit(dt_norm[cols])

        new_values = self.pca.transform(dt_norm[cols])

        for comp in range(0, number_comp):
            data_table["pca_" + str(comp + 1)] = new_values[:, comp]

        return data_table
