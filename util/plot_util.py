import matplotlib.pyplot as plt
from util.file_util import FileUtil
from business.metrics_functions import MetricsFunctions
from util.constants import Constants
import numpy as np
from pandas.plotting import table


class PlotUtil:

    @staticmethod
    def generate_plot(title: str, data):
        fig, ax = plt.subplots()
        ax.set_title(title)
        plt.plot(data)
        filename = title.lower().replace(' ', '_')
        plt.savefig("plots/" + filename + ".png")
        plt.close(fig)

    @staticmethod
    def generate_table_by_dataframe(df):
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.set_title("Mean Metrics Functions")
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)
        tab = table(ax, df, loc='upper right')
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        plt.savefig("plots/mean_metrics_functions.png")
        plt.close(fig)

    @staticmethod
    def get_values_to_print_mean_metrics(filename_array):
        file_array_read = PlotUtil.get_values_organized_by_metric(filename_array)
        array_mean = []
        for metric in MetricsFunctions.ALL_METRICS:
            array_by_metric = PlotUtil.organize_data_by_metric(file_array_read, metric)
            for value_metric in array_by_metric:
                array_mean.append(np.sum(value_metric)/Constants.N_ITERATIONS)
        return array_mean

    @staticmethod
    def get_values_organized_by_metric(filename_array):
        file_array_read = []
        for filename in filename_array:
            file_array_read.append(FileUtil.read_file(filename=filename))
        return file_array_read

    @staticmethod
    def generate_boxplot_by_file(filename_array):
        file_array_read = PlotUtil.get_values_organized_by_metric(filename_array)
        PlotUtil.generate_boxplot(data=file_array_read, filename_array=filename_array)

    @staticmethod
    def generate_boxplot(data, filename_array):
        for metric in MetricsFunctions.ALL_METRICS:
            fig, ax = plt.subplots()
            ax.set_title('Boxplot ' + metric)
            ax.boxplot(PlotUtil.organize_data_by_metric(data, metric))
            plt.xticks(PlotUtil.array_index_boxplot(filename_array), filename_array)
            plt.savefig("plots/boxplot_"+metric+".png")
            plt.close(fig)

    @staticmethod
    def array_index_boxplot(array):
        index_array = []
        for i in range(len(array)):
            index_array.append(i+1)
        return index_array

    @staticmethod
    def organize_data_by_metric(array, metric):
        organized_array_boxplot = []
        for data in array:
            array_function = []
            for metric_data in data:
                if metric in metric_data:
                    values = (metric_data.split(":")[1]).split("|")
                    values.pop()
                    for value in values:
                        array_function.append(float(value))
            organized_array_boxplot.append(array_function)
        return organized_array_boxplot

