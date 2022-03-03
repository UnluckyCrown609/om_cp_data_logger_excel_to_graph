from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from pathlib import Path
from scipy import signal


class OmegaLogger:

    def __init__(self, name, start_col='Start', end_col='End'):
        self.name = name
        self.start_col = start_col
        self.end_col = end_col

        self.data = self.get_data()
        self.device_info = self.get_device_info()

    def get_data(self):
        self.data = pd.read_excel(f'../spreadsheets/impact_{self.name}.xlsx',
                                  skiprows=6, parse_dates=True, index_col='Time')
        self.data.drop(['Date'], axis=1, inplace=True)
        return self.data

    def get_device_info(self):
        self.device_info = pd.read_excel(f'../spreadsheets/impact_{self.name}.xlsx', skipfooter=248641)
        self.device_info = self.device_info.drop(['Unnamed: 1'], axis=1)
        self.device_info = self.device_info.T.reset_index()
        self.device_info.drop(range(2, len(self.device_info.index), 1), axis=0, inplace=True)
        self.device_info.columns = self.device_info.iloc[0]
        self.device_info = self.device_info.drop([0], axis=0).reset_index(drop=True)
        return self.device_info

    def get_test_times(self):
        test_times = pd.read_json(f'../json_tables/times_{self.name}.json',
                                  convert_dates=(self.start_col, self.end_col))
        return test_times

    def get_split_data(self):
        dfs = {}

        for x in range(len(self.get_test_times())):
            dfs[x] = self.data[self.get_test_times()[self.start_col][x]:self.get_test_times()[self.end_col][x]]
        return dfs

    @staticmethod
    def resample(df, col_name, order, cut_off, b_type):
        new_dfs = {}
        return_dfs = {}
        if type(df) == dict:
            for x in range(len(df)):
                b, a = signal.butter(order, cut_off, btype=b_type, analog=False)
                new_dfs[x] = pd.DataFrame(signal.filtfilt(b, a, df[x][col_name]))
                new_dfs[x].index = df[x].index
                return_dfs[x] = pd.concat([df[x], new_dfs[x]], axis=1)
                return_dfs[x].rename(columns={0: f"{col_name}_resample"}, inplace=True)
            return return_dfs
        else:
            b, a = signal.butter(order, cut_off, btype=b_type, analog=False)
            new_dfs = pd.DataFrame(signal.filtfilt(b, a, df[col_name]))
            new_dfs.index = df.index
            return_dfs = pd.concat([df, new_dfs], axis=1)
            return_dfs.rename(columns={0: f"{col_name}_resample"}, inplace=True)
        return return_dfs

    def graphing(self, df, name_addon=""):
        p = Path(f'../graphs/{self.name}')
        if type(df) == dict:
            for x in range(len(df)):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df[x]['Shock - Z Axis (g)'], color='#ffb300', marker='o', markersize=1, label='Test 5')
                ax.set_xlabel('Time (Hours:Minutes:Seconds.Milliseconds)', fontsize=15)
                ax.set_ylabel('Acceleration (g)', fontsize=15)
                plt.xticks(rotation=15)
                plt.grid()
                date_form = DateFormatter("%H:%M:%S.%f")
                ax.xaxis.set_major_formatter(date_form)
                self.annot_max(df[x].index, df[x]['Shock - Z Axis (g)'])
                self.annot_min(df[x].index, df[x]['Shock - Z Axis (g)'])
                p.mkdir(exist_ok=True)
                # plt.tight_layout()
                fig.set_size_inches(12, 8)
                fig.savefig(f'../graphs/{self.name}/impact_{self.name}_test_{x}{name_addon}.pdf')
                plt.close(fig)

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df['Shock - Z Axis (g)'], color='#ffb300', marker='o', markersize=1, label='Test 5')
            ax.set_xlabel('Time (Hours:Minutes:Seconds)', fontsize=15)
            ax.set_ylabel('Acceleration (g)', fontsize=15)
            plt.xticks(rotation=15)
            plt.grid()
            date_form = DateFormatter("%H:%M:%S")
            ax.xaxis.set_major_formatter(date_form)
            self.annot_max(df.index, df['Shock - Z Axis (g)'])
            self.annot_min(df.index, df['Shock - Z Axis (g)'])
            p.mkdir(exist_ok=True)
            # plt.tight_layout()
            fig.set_size_inches(12, 8)
            fig.savefig(f'../graphs/{self.name}/impact_{self.name}_test{name_addon}.pdf')
            plt.close(fig)

    def graphing_2_dataframes(self, df1, df2, name_addon=""):
        p = Path(f'../graphs/{self.name}')
        if type(df1) == dict:
            for x in range(len(df1)):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df1[x]['Shock - Z Axis (g)'], color='#ffb300', marker='o', markersize=1, label='Original')
                ax.plot(df2[x]['Shock - Z Axis (g)_resample'], color='#00c8ff', marker='o', markersize=1,
                        label='Resample')
                ax.set_xlabel('Time (Hours:Minutes:Seconds.Milliseconds)', fontsize=15)
                ax.set_ylabel('Acceleration (g)', fontsize=15)
                plt.xticks(rotation=15)
                plt.grid()
                ax.legend()
                date_form = DateFormatter("%H:%M:%S.%f")
                ax.xaxis.set_major_formatter(date_form)
                self.annot_max(df1[x].index, df1[x]['Shock - Z Axis (g)'])
                self.annot_min(df1[x].index, df1[x]['Shock - Z Axis (g)'])
                p.mkdir(exist_ok=True)
                # plt.tight_layout()
                fig.set_size_inches(12, 8)
                fig.savefig(f'../graphs/{self.name}/impact_{self.name}_test_{x}{name_addon}.pdf')
                plt.close(fig)

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df1['Shock - Z Axis (g)'], color='#ffb300', marker='o', markersize=1, label='Original')
            ax.plot(df2['Shock - Z Axis (g)_resample'], color='#00c8ff', marker='o', markersize=1, label='Resample')
            ax.set_xlabel('Time (Hours:Minutes:Seconds)', fontsize=15)
            ax.set_ylabel('Acceleration (g)', fontsize=15)
            plt.xticks(rotation=15)
            plt.grid()
            ax.legend()
            date_form = DateFormatter("%H:%M:%S")
            ax.xaxis.set_major_formatter(date_form)
            self.annot_max(df1.index, df1['Shock - Z Axis (g)'])
            self.annot_min(df1.index, df1['Shock - Z Axis (g)'])
            p.mkdir(exist_ok=True)
            # plt.tight_layout()
            fig.set_size_inches(12, 8)
            fig.savefig(f'../graphs/{self.name}/impact_{self.name}_test{name_addon}.pdf')
            plt.close(fig)

    @staticmethod
    def annot_max(x, y, ax=None):
        x_max = x[np.argmax(y)]
        y_max = y.max()
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrow_props = dict(arrowstyle="->")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrow_props, bbox=bbox_props, ha="left", va="top")
        ax.annotate(f"Max={y_max}", xy=(x_max, y_max), xytext=(0.04, 0.94), **kw)

    @staticmethod
    def annot_min(x, y, ax=None):
        x_min = x[np.argmin(y)]
        y_min = y.min()
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrow_props = dict(arrowstyle="->")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrow_props, bbox=bbox_props, ha="left", va="top")
        ax.annotate(f"Min={y_min}", xy=(x_min, y_min), xytext=(0.80, 0.10), **kw)

    @staticmethod
    def find_max_and_min(df, col_name_x, col_name_y):
        if type(df) == dict:
            min = {}
            max = {}
            for k in range(len(df)):
                tmp = df[k].reset_index()
                x = tmp[col_name_x]
                y = tmp[col_name_y]
                min[k] = (x[np.argmin(y)], y.min())
                max[k] = (x[np.argmax(y)], y.max())
            return min, max
        else:
            tmp = df.reset_index()
            x = tmp[col_name_x]
            y = tmp[col_name_y]
            return (x[np.argmin(y)], y.min()), (x[np.argmax(y)], y.max())

    def auto_split_data(self, df, time):
        dfs = {}
        min = {}
        max = {}
        x_start = {}
        x_end = {}
        dfu = {}

        for k in range(len(self.get_test_times())):
            dfs[k] = df[self.get_test_times()[self.start_col][k]:self.get_test_times()[self.end_col][k]]
            min[k], max[k] = self.find_max_and_min(dfs[k], "Time", "Shock - Z Axis (g)")
            x_start[k] = min[k][0] - timedelta(milliseconds=time)
            x_end[k] = min[k][0] + timedelta(milliseconds=time)
            dfu[k] = df[x_start[k]:x_end[k]]
        return dfu

    @staticmethod
    def intergrate(x_array1, y_array2):
        tmp = float(np.trapz(abs(x_array1), y_array2))
        return tmp * (10 ** -9)

    @staticmethod
    def to_latex(df, filename, caption, label):
        df.to_latex(f'../tex_files/{filename}.tex', longtable=True,
                    index=False, caption=caption, label=label)
        df.to_latex(f'../tex_files/{filename}_escape.tex', longtable=True,
                    index=False, caption=caption, label=label, escape=False)
        return


for item, cut_off in zip(['hard'], [0.15]):
    OME = OmegaLogger(item, 'Start', 'End')
    data = OME.auto_split_data(OME.data, 200)
    resample_data = OME.resample(data, 'Shock - Z Axis (g)', 5, cut_off, "lowpass")
    OME.graphing_2_dataframes(data, resample_data)

    z_min, z_max = OME.find_max_and_min(resample_data, "Time", "Shock - Z Axis (g)")

    report_table = pd.DataFrame(columns=['ID', 'Time of impact',
                                         r'Peak Force (\unit{\gram})',
                                         r'Speed at point of impact (\unit{\metre\per\second})'])

    for x in data:
        report_table = report_table.append({
            'ID': f"Test {x + 1}",
            'Time of impact': z_min[x][0],
            r'Peak Force (\unit{\gram})': z_min[x][1],
            r'Speed at point of impact (\unit{\metre\per\second})': OME.intergrate(
                data[x]["Shock - Z Axis (g)"].values,
                data[x].index.values)}, ignore_index=True)

    for x in ['Single Fault Condition Insertions']:
        OME.to_latex(report_table, f"{OME.name}", f"Summary of {x}", f"{OME.name}")
