# Import relevant libraries
from torch import from_numpy, unsqueeze
import torch
from scipy.signal import find_peaks


try:
    import numpy as np
    from pandas import read_csv, DataFrame
    import matplotlib.pyplot as plt
    import sys

    # import os
    # from tensorflow.python.ops.numpy_ops import np_config
    # from custom_loss import class_loss

    # np_config.enable_numpy_behavior()

    no_of_bins = 1500
    label = ["Am", "Bg", "Co", "Cs"]


    def check_low_flat_spectrum(spectrum, value_threshold, flatness_threshold):
        """
        Checks if the spectrum has low and flat values.

        Args:
            spectrum (pandas.Series): The spectrum data.
            value_threshold (float): The threshold for the values to be considered low.
            flatness_threshold (float): The maximum allowed difference between max and min values.

        Returns:
            bool: True if the spectrum is low and flat, False otherwise.
        """
        max_value = spectrum[1:].max()
        min_value = spectrum[1:].min()
        # print(max_value,(max_value - min_value))
        return max_value < value_threshold and (max_value - min_value) < flatness_threshold


    def plot_spectrum_with_peaks(data, title,peaks_ht):
        """
        Plots the spectrum data and highlights the peaks identified.
        Also shows the height threshold used for peak finding.
        """
        spectrum = np.array(data.iloc[:, 1])
        bins = np.array(data.iloc[:, 0])

        # Find peaks
        peaks, properties = find_peaks(spectrum, height=peaks_ht)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(bins, spectrum, label='Spectrum')
        plt.scatter(bins[peaks], spectrum[peaks], color='red', label='Peaks')
        plt.axhline(y=peaks_ht, color='green', linestyle='--', label='Height Threshold (0.1)')
        plt.title(title)
        plt.xlabel('Bin Number')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


    def normalize(arr):
        """
        Normalizes an array by dividing each element by the sum of all elements.

        Args:
            arr (list): A list of numbers to normalize.

        Returns:
            list: A normalized list of numbers.
        """
        arr = [float(x) for x in arr]
        arr_sum = sum(arr)
        return [float(i) / arr_sum for i in arr]

    def main():
        # Load the trained models
        # class_model = load("cnn_class.joblib")
        # conc_model = load("cnn_conc.joblib")
        folder_name  = "remtec_david_models"
        L_class_model = torch.jit.load(f"class_model.pt")
        L_class_model.eval()

        L_conc_model = torch.jit.load(f"conc_model.pt")
        L_conc_model.eval()

        # class_model.compile(loss=class_loss)

        # Load the spectrum File

        spectrum_data = read_csv(
            "IsoSpectrum.csv",
            # "C:/theCave/ISO-ID/Remtec_captures/jan30/captured_spectrums/Bg_8mn.csv",
            # "C:/theCave/ISO-ID/Remtec_captures/jan30/Captured_files/Captured_files/cs_captures/Cs_coo1_2.csv",
            # "C:/theCave/ISO-ID/spectrum_corrector/remtec_exec/cs_bg/IsoSpectrum.csv",
            header=None,
        )

        spectrum = spectrum_data.iloc[:, 1]

        #prr checks for sanity of spectrums
        # Define thresholds
        value_threshold = 1e-6      # Adjust based on your data
        flatness_threshold = 1e-5   # Adjust based on your data

        if check_low_flat_spectrum(spectrum, value_threshold, flatness_threshold):
            # print("The spectrum has low and flat values.")
            return  # Exit if the spectrum is low and flat

        # spectrum = spectrum.iloc[0,1:1501]
        peaks_ht = 0.2
        peaks, _ = find_peaks(spectrum, height=peaks_ht)
        # plot_spectrum_with_peaks(spectrum_data, "title",peaks_ht)
        spectrum = normalize(spectrum)
        spectrum = np.array(spectrum, dtype=np.float32).reshape(-1, 1500)
        cobalt_region = spectrum[0, 900:1400].sum()
        cesium_region = spectrum[0, 650:674].sum()
        spectrum = (spectrum * 100000).astype(np.float32)
        

        # print("Peaks : ", peaks)

        class_predictions = np.round(
            L_class_model(unsqueeze(from_numpy(spectrum), 0)).detach(), 3
        )
        conc_predictions = np.round(
            L_conc_model(unsqueeze(from_numpy(spectrum), 0)).detach(), 3
        )
        threshold = 0.9
        conc_threshold = 0.1

        # for i, pred in enumerate(conc_predictions[0]):
        #     print(f"i in {i} and pred : {pred}")
        #     if pred < 0.1:
        #         class_predictions[0, i] = 0

        if class_predictions[0, 2] > threshold and not (cobalt_region > 0.09):
            class_predictions[0, 2] = 0.0


        cs_peak_range = (650, 670)
        cs_present = any(cs_peak_range[0] <= peak <= cs_peak_range[1] for peak in peaks)
        


        # print(f"class: ", class_predictions)
        # print(f"conc: ", conc_predictions)
        # print("======================================================================")

        # if class_predictions[0, 3] > threshold and not cs_present:
        #     class_predictions[0, 3] = 0.0

        if class_predictions[0, 3] > threshold and not (cesium_region > 0.015):
            class_predictions[0, 3] = 0.0


        # print("sum: ", class_predictions+conc_predictions)

        # Create CSV file
        predictions = np.where(class_predictions > threshold)[1]
        filtered_predictions = []
        for ind in predictions:
            if conc_predictions[0, ind] > conc_threshold:
                filtered_predictions.append(ind)
        isotope_name = [label[idx] for idx in filtered_predictions]
        conc_values = [conc_predictions[0, idx] for idx in filtered_predictions]

        # print("Iso names ", isotope_name)
        # print("values ", conc_predictions)
        result = []
        for i in range(len(isotope_name)):
            result.append([isotope_name[i], np.round(float(conc_values[i]), 2)])
        if len(result) == 0:
            result = [["Bg", 1]]
        df = DataFrame(result)
        df.to_csv(
            "IsoResults.csv",
            index=False,
            header=False,
        )

except Exception as e:
    print("Error occured : ", e)

main()


# if sys.stdin.isatty():
#     input("Press Enter to continue...")
