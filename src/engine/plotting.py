"""绘制异常分数、阈值和检测结果。"""


import pandas as pd

import numpy as np

import os

import json

from datetime import datetime

import plotly as py

import matplotlib.pyplot as plt

import plotly.graph_objs as go

from plotly.subplots import make_subplots

import cufflinks as cf

import seaborn as sns

import matplotlib.pyplot as plt


from src.data.utils import get_data_dim, get_series_color, get_y_height
cf.go_offline()


class Plotter:


    """

    用于可视化异常检测结果的类。

    Includes visualization of forecasts, reconstructions, anomaly scores, predicted and actual anomalies

    Plotter-class inspired by TelemAnom (https://github.com/khundman/telemanom)

    """


    def __init__(self, result_path, model_id='-1'):

        self.result_path = result_path

        self.model_id = model_id

        self.train_output = None

        self.test_output = None

        self.labels_available = True

        self.pred_cols = None

        self._load_results()

        self.train_output["timestamp"] = self.train_output.index

        self.test_output["timestamp"] = self.test_output.index


        config_path = f"{self.result_path}/config.txt"

        with open(config_path) as f:

            self.lookback = json.load(f)["lookback"]


        if "SMD" in self.result_path:

            self.pred_cols = [f"feat_{i}" for i in range(get_data_dim("machine"))]

        elif "SMAP" in self.result_path or "MSL" in self.result_path:

            self.pred_cols = ["feat_1"]

        elif "CALCE" in self.result_path:

            self.pred_cols = ["capacity"]

        elif "BMS" in self.result_path:

            self.pred_cols = ["SYS_Vol", "SYS_I", "SYS_DSOC", "SYS_SOH", "SYS_Vmax"]


    def _load_results(self):

        if self.model_id.startswith('-'):

            # 如果传入的是相对模型编号，自动解析最新结果目录

            if 'universal_model' in self.result_path:

                # 对于通用模型，在 universal_model 文件夹内查找日期时间目录

                universal_path = f"{self.result_path}"

                dir_content = os.listdir(universal_path)

                datetimes = []

                for subf in dir_content:

                    # 跳过 logs 等非结果目录

                    if os.path.isdir(f"{universal_path}/{subf}") and subf not in ['logs']:

                        try:

                            dt = datetime.strptime(subf, '%d%m%Y_%H%M%S')

                            datetimes.append(dt)

                        except ValueError:

                            # 跳过无法解析为日期时间的目录

                            continue


                if not datetimes:

                    # 如果没有日期时间目录，检查当前路径是否已经是有效结果目录

                    # universal_model 路径本身也可能直接包含结果文件

                    parent_dir = os.path.dirname(self.result_path.rstrip('/\\'))

                    if 'universal_model' in parent_dir:

                        # 当前已经位于 universal_model 结果目录中

                        pass  # self.result_path 已经正确

                    else:

                        raise ValueError(f"No valid datetime directories found in {universal_path}")

                else:

                    datetimes.sort()

                    model_id = datetimes[int(self.model_id)].strftime('%d%m%Y_%H%M%S')

                    self.result_path = f'{universal_path}/{model_id}'

            else:

                # 普通模型目录

                dir_content = os.listdir(self.result_path)

                datetimes = []

                for subf in dir_content:

                    # 跳过 universal_model 和 logs 等非日期时间目录

                    if os.path.isdir(f"{self.result_path}/{subf}") and subf not in ['logs']:

                        try:

                            dt = datetime.strptime(subf, '%d%m%Y_%H%M%S')

                            datetimes.append(dt)

                        except ValueError:

                            # 跳过无法解析为日期时间的目录

                            continue


                if not datetimes:

                    # 检查当前路径是否已经是日期时间目录

                    path_leaf = os.path.basename(self.result_path.rstrip('/\\'))

                    # 仅当 path_leaf 看起来像日期时间字符串时才尝试解析

                    if '_' in path_leaf and path_leaf.replace('_', '').isdigit():

                        try:

                            datetime.strptime(path_leaf, '%d%m%Y_%H%M%S')

                            # 当前已经位于日期时间目录中，无需修改 self.result_path

                        except ValueError:

                            # 检查当前目录下是否存在 universal_model 子目录

                            if os.path.exists(os.path.join(self.result_path, 'universal_model')):

                                # 切换到 universal_model 目录

                                self.result_path = os.path.join(self.result_path, 'universal_model')

                                # 递归加载实际结果目录

                                self._load_results()

                                return

                            else:

                                raise ValueError(f"No valid datetime directories found in {self.result_path}")

                    else:

                        # 检查当前目录下是否存在 universal_model 子目录

                        if os.path.exists(os.path.join(self.result_path, 'universal_model')):

                            # 切换到 universal_model 目录

                            self.result_path = os.path.join(self.result_path, 'universal_model')

                            # 递归加载实际结果目录

                            self._load_results()

                            return

                        else:

                            raise ValueError(f"No valid datetime directories found in {self.result_path}")

                else:

                    datetimes.sort()

                    model_id = datetimes[int(self.model_id)].strftime('%d%m%Y_%H%M%S')

                    self.result_path = f'{self.result_path}/{model_id}'


        print(f"Loading results of {self.result_path}")


        # 处理通用模型的实体子目录

        if 'universal_model' in self.result_path and not self.model_id.startswith('-'):

            # 选择第一个实体目录加载结果

            entity_dirs = [d for d in os.listdir(self.result_path)

                          if os.path.isdir(os.path.join(self.result_path, d)) and (d.isdigit() or d.startswith('Cell'))]

            if entity_dirs:

                # 按实体编号排序

                # 兼容纯数字实体名

                if all(d.isdigit() for d in entity_dirs):

                    first_entity = sorted(entity_dirs, key=int)[0]

                else:

                    # 按 Cell 编号排序

                    first_entity = sorted(entity_dirs, key=lambda x: int(x.replace('Cell', '')) if x.startswith('Cell') else int(x))[0]

                result_dir = os.path.join(self.result_path, first_entity)

                print(f"Loading results from entity {first_entity} directory: {result_dir}")

            else:

                result_dir = self.result_path

        # model_id 为相对编号时同样处理通用模型目录

        elif 'universal_model' in self.result_path and self.model_id.startswith('-'):

            # 选择第一个实体目录加载结果

            entity_dirs = [d for d in os.listdir(self.result_path)

                          if os.path.isdir(os.path.join(self.result_path, d)) and (d.isdigit() or d.startswith('Cell'))]

            if entity_dirs:

                # 按实体编号排序

                # 兼容纯数字实体名

                if all(d.isdigit() for d in entity_dirs):

                    first_entity = sorted(entity_dirs, key=int)[0]

                else:

                    # 按 Cell 编号排序

                    first_entity = sorted(entity_dirs, key=lambda x: int(x.replace('Cell', '')) if x.startswith('Cell') else int(x))[0]

                result_dir = os.path.join(self.result_path, first_entity)

                print(f"Loading results from entity {first_entity} directory: {result_dir}")

            else:

                result_dir = self.result_path

        else:

            result_dir = self.result_path


        train_output = pd.read_pickle(f"{result_dir}/train_output.pkl")

        train_output.to_pickle(f"{result_dir}/train_output.pkl")

        train_output["A_True_Global"] = 0

        test_output = pd.read_pickle(f"{result_dir}/test_output.pkl")


        # SMAP 和 MSL 只预测一个特征

        if 'SMAP' in self.result_path or 'MSL' in self.result_path:

            train_output[f'A_Pred_0'] = train_output['A_Pred_Global']

            train_output[f'A_Score_0'] = train_output['A_Score_Global']

            train_output[f'Thresh_0'] = train_output['Thresh_Global']


            test_output[f'A_Pred_0'] = test_output['A_Pred_Global']

            test_output[f'A_Score_0'] = test_output['A_Score_Global']

            test_output[f'Thresh_0'] = test_output['Thresh_Global']

        # 处理 CALCE 和 CALCE2 数据集

        elif 'CALCE' in self.result_path:

            train_output[f'A_Pred_0'] = train_output['A_Pred_Global']

            train_output[f'A_Score_0'] = train_output['A_Score_Global']

            train_output[f'Thresh_0'] = train_output['Thresh_Global']


            test_output[f'A_Pred_0'] = test_output['A_Pred_Global']

            test_output[f'A_Score_0'] = test_output['A_Score_Global']

            test_output[f'Thresh_0'] = test_output['Thresh_Global']

        # 处理 BMS 数据集

        elif 'BMS' in self.result_path:

            # BMS 只有全局异常分数，复制到每个特征列用于绘图

            for i in range(get_data_dim("BMS")):  # BMS 当前有 5 个特征

                train_output[f'A_Pred_{i}'] = train_output['A_Pred_Global']

                train_output[f'A_Score_{i}'] = train_output['A_Score_Global']

                train_output[f'Thresh_{i}'] = train_output['Thresh_Global']


                test_output[f'A_Pred_{i}'] = test_output['A_Pred_Global']

                test_output[f'A_Score_{i}'] = test_output['A_Score_Global']

                test_output[f'Thresh_{i}'] = test_output['Thresh_Global']


        self.train_output = train_output

        self.test_output = test_output


    def result_summary(self):

        if 'SMAP' in self.result_path or 'MSL' in self.result_path or 'SMD' in self.result_path:

            # 标准数据集的 summary 直接位于结果目录

            path = f"{self.result_path}/summary.txt"

        elif 'CALCE' in self.result_path or 'BMS' in self.result_path or 'NASA' in self.result_path:

            # 电池类数据集可能使用通用模型目录结构

            if 'universal_model' in self.result_path:

                # 对于通用模型，summary 可能位于父目录或实体目录中

                parent_summary = f"{self.result_path}/summary.txt"

                if os.path.exists(parent_summary):

                    path = parent_summary

                else:

                    # 在实体目录中查找 summary

                    entity_dirs = [d for d in os.listdir(self.result_path)

                                  if os.path.isdir(os.path.join(self.result_path, d)) and (d.isdigit() or d.startswith('Cell'))]

                    if entity_dirs:

                        # 使用第一个实体的 summary

                        if all(d.isdigit() for d in entity_dirs):

                            first_entity = sorted(entity_dirs, key=int)[0]

                        else:

                            first_entity = sorted(entity_dirs, key=lambda x: int(x.replace('Cell', '')) if x.startswith('Cell') else int(x))[0]

                        path = f"{self.result_path}/{first_entity}/summary.txt"

                    else:

                        path = parent_summary  # 回退到父目录 summary

            else:

                path = f"{self.result_path}/summary.txt"

        else:

            path = f"{self.result_path}/summary.txt"


        if not os.path.exists(path):

            print(f"Summary file not found at {path}")

            return


        with open(path) as f:

            summary = json.load(f)


        print("------------------------------------------")

        print("------------ Evaluation Summary -----------")

        print("------------------------------------------")


        for key, value in summary.items():

            print(f"{key}:")

            for k, v in value.items():

                print(f"\t{k}: {v}")

            print()


    def create_shapes(self, ranges, sequence_type, _min, _max, plot_values, is_test=True, xref=None, yref=None):

        """

        为 plotly 中需要高亮的区域创建形状（真实和预测异常片段）。


        :param ranges: tuple of start and end indices for anomaly sequences for a feature

        :param sequence_type: "predict" if predicted values else "true" if actual values. Determines colors.

        :param _min: min y value of series

        :param _max: max y value of series

        :param plot_values: dictionary of different series to be plotted


        :return: list of shapes specifications for plotly

        """


        if _max is None:

            _max = max(plot_values["errors"])


        if sequence_type is None:

            color = "blue"

        else:

            color = "red" if sequence_type == "true" else "blue"

        shapes = []


        for r in ranges:

            w = 5

            x0 = r[0] - w

            x1 = r[1] + w

            shape = {

                "type": "rect",

                "x0": x0,

                "y0": _min,

                "x1": x1,

                "y1": _max,

                "fillcolor": color,

                "opacity": 0.08,

                "line": {

                    "width": 0,

                },

            }

            if xref is not None:

                shape["xref"] = xref

                shape["yref"] = yref


            shapes.append(shape)


        return shapes


    @staticmethod

    def get_anomaly_sequences(values):

        splits = np.where(values[1:] != values[:-1])[0] + 1

        if values[0] == 1:

            splits = np.insert(splits, 0, 0)


        a_seqs = []

        for i in range(0, len(splits) - 1, 2):

            a_seqs.append([splits[i], splits[i + 1] - 1])


        if len(splits) % 2 == 1:

            a_seqs.append([splits[-1], len(values) - 1])


        return a_seqs


    def plot_train_test_errors(self, plot_train=False):

        """

        绘制所有特征的训练/测试真实值和预测值，图表保存到结果目录。

        :param plot_train: If true, plot training data. If false, plot test data

        """


        if plot_train:

            output = self.train_output

            title_prefix = "Train"

        else:

            output = self.test_output

            title_prefix = "Test"


        for p_col in self.pred_cols:

            y_true = output[p_col]

            y_pred = output[f'{p_col}|Pred']


            trace_true = go.Scatter(

                x=output["timestamp"],

                y=y_true,

                mode="lines",

                name="True",

                line=dict(color='blue')

            )


            trace_pred = go.Scatter(

                x=output["timestamp"],

                y=y_pred,

                mode="lines",

                name="Predicted",

                line=dict(color='red')

            )


            layout = go.Layout(

                title=f"{title_prefix} {p_col} Predictions",

                xaxis=dict(title="Timestamp"),

                yaxis=dict(title=p_col),

                hovermode='closest',

                showlegend=True

            )


            fig = go.Figure(data=[trace_true, trace_pred], layout=layout)


            # 保存图表

            save_path = f"{self.result_path}/{title_prefix.lower()}_{p_col}_predictions.html"

            py.offline.plot(fig, filename=save_path, auto_open=False)

            print(f"{title_prefix} {p_col} predictions plot saved to {save_path}")


    def plot_errors(self, plot_train=False):

        """

        绘制所有特征的异常点和异常分数，图表保存到结果目录。

        :param plot_train: If true, plot training data. If false, plot test data

        """


        if plot_train:

            output = self.train_output

            title_prefix = "Train"

        else:

            output = self.test_output

            title_prefix = "Test"


        for p_col in self.pred_cols:

            # 异常分数曲线

            trace_score = go.Scatter(

                x=output["timestamp"],

                y=output[f"A_Score_{p_col.split('_')[1] if '_' in p_col else '0'}"],

                mode="lines",

                name="Anomaly Score",

                line=dict(color='blue')

            )


            # 阈值曲线

            trace_thresh = go.Scatter(

                x=output["timestamp"],

                y=output[f"Thresh_{p_col.split('_')[1] if '_' in p_col else '0'}"],

                mode="lines",

                name="Threshold",

                line=dict(color='red')

            )


            layout = go.Layout(

                title=f"{title_prefix} {p_col} Anomaly Scores",

                xaxis=dict(title="Timestamp"),

                yaxis=dict(title="Score"),

                hovermode='closest',

                showlegend=True

            )


            fig = go.Figure(data=[trace_score, trace_thresh], layout=layout)


            # 保存图表

            save_path = f"{self.result_path}/{title_prefix.lower()}_{p_col}_anomaly_scores.html"

            py.offline.plot(fig, filename=save_path, auto_open=False)

            print(f"{title_prefix} {p_col} anomaly scores plot saved to {save_path}")


            # 预测异常点

            trace_pred_anom = go.Scatter(

                x=output["timestamp"],

                y=output[f"A_Pred_{p_col.split('_')[1] if '_' in p_col else '0'}"],

                mode="markers",

                name="Predicted Anomalies",

                marker=dict(color='red', size=8)

            )


            # 如果有真实标签，则同时绘制真实异常点

            if self.labels_available:

                trace_true_anom = go.Scatter(

                    x=output["timestamp"],

                    y=output[f"A_True_{p_col.split('_')[1] if '_' in p_col else '0'}"],

                    mode="markers",

                    name="True Anomalies",

                    marker=dict(color='orange', size=8)

                )

                data = [trace_pred_anom, trace_true_anom]

            else:

                data = [trace_pred_anom]


            layout = go.Layout(

                title=f"{title_prefix} {p_col} Anomalies",

                xaxis=dict(title="Timestamp"),

                yaxis=dict(title="Anomaly"),

                hovermode='closest',

                showlegend=True

            )


            fig = go.Figure(data=data, layout=layout)


            # 保存图表

            save_path = f"{self.result_path}/{title_prefix.lower()}_{p_col}_anomalies.html"

            py.offline.plot(fig, filename=save_path, auto_open=False)

            print(f"{title_prefix} {p_col} anomalies plot saved to {save_path}")


    def plot_global_predictions(self, type="test"):

        """

        绘制全局预测、异常分数和异常点。

        :param type: "train" or "test"

        """

        if type == "train":

            output = self.train_output

            title_prefix = "Train"

        else:

            output = self.test_output

            title_prefix = "Test"


        # 创建子图

        fig = make_subplots(rows=2, cols=1, subplot_titles=(f"{title_prefix} Global Predictions", f"{title_prefix} Global Anomaly Scores"))


        # 绘制所有特征的真实值和预测值

        for p_col in self.pred_cols:

            fig.add_trace(

                go.Scatter(

                    x=output["timestamp"],

                    y=output[p_col],

                    mode="lines",

                    name=f"{p_col} True",

                    line=dict(color='blue')

                ),

                row=1, col=1

            )


            fig.add_trace(

                go.Scatter(

                    x=output["timestamp"],

                    y=output[f'{p_col}|Pred'],

                    mode="lines",

                    name=f"{p_col} Predicted",

                    line=dict(color='red')

                ),

                row=1, col=1

            )


        # 绘制全局异常分数

        fig.add_trace(

            go.Scatter(

                x=output["timestamp"],

                y=output["A_Score_Global"],

                mode="lines",

                name="Global Anomaly Score",

                line=dict(color='blue')

            ),

            row=2, col=1

        )


        fig.add_trace(

            go.Scatter(

                x=output["timestamp"],

                y=output["Thresh_Global"],

                mode="lines",

                name="Threshold",

                line=dict(color='red')

            ),

            row=2, col=1

        )


        fig.update_xaxes(title_text="Timestamp", row=1, col=1)

        fig.update_xaxes(title_text="Timestamp", row=2, col=1)

        fig.update_yaxes(title_text="Value", row=1, col=1)

        fig.update_yaxes(title_text="Score", row=2, col=1)

        fig.update_layout(height=800, title_text=f"{title_prefix} Global Results")


        # 保存图表

        save_path = f"{self.result_path}/{title_prefix.lower()}_global_results.html"

        py.offline.plot(fig, filename=save_path, auto_open=False)

        print(f"{title_prefix} global results plot saved to {save_path}")


        # 绘制异常点

        fig_anom = go.Figure()


        # 预测异常点

        fig_anom.add_trace(

            go.Scatter(

                x=output["timestamp"],

                y=output["A_Pred_Global"],

                mode="markers",

                name="Predicted Anomalies",

                marker=dict(color='red', size=8)

            )

        )


        # 如果有真实标签，则同时绘制真实异常点

        if self.labels_available:

            fig_anom.add_trace(

                go.Scatter(

                    x=output["timestamp"],

                    y=output["A_True_Global"],

                    mode="markers",

                    name="True Anomalies",

                    marker=dict(color='orange', size=8)

                )

            )


        fig_anom.update_layout(

            title=f"{title_prefix} Global Anomalies",

            xaxis=dict(title="Timestamp"),

            yaxis=dict(title="Anomaly"),

            hovermode='closest',

            showlegend=True

        )


        # 保存图表

        save_path = f"{self.result_path}/{title_prefix.lower()}_global_anomalies.html"

        py.offline.plot(fig_anom, filename=save_path, auto_open=False)

        print(f"{title_prefix} global anomalies plot saved to {save_path}")


    def plot_feature(self, feature=0, plot_train=False, plot_errors=False, plot_feature_anom=False, start=0, end=-1):

        """

        绘制单个特征。

        :param feature: feature index

        :param plot_train: If true, plot training data. If false, plot test data

        :param plot_errors: If true, plot anomaly scores

        :param plot_feature_anom: If true, plot feature-level anomalies

        :param start: start index

        :param end: end index

        """

        if plot_train:

            output = self.train_output

            title_prefix = "Train"

        else:

            output = self.test_output

            title_prefix = "Test"


        if end == -1:

            end = len(output)


        output = output.iloc[start:end]


        if 'SMAP' in self.result_path or 'MSL' in self.result_path:

            p_col = "feat_1"

        elif "SMD" in self.result_path:

            p_col = f"feat_{feature}"

        elif 'CALCE' in self.result_path:

            p_col = "capacity"

        elif 'BMS' in self.result_path:

            # BMS 有 5 个特征：SYS_Vol、SYS_I、SYS_DSOC、SYS_SOH、SYS_Vmax

            bms_features = ["SYS_Vol", "SYS_I", "SYS_DSOC", "SYS_SOH", "SYS_Vmax"]

            p_col = bms_features[feature] if feature < len(bms_features) else bms_features[0]

        else:  # 默认特征列

            p_col = f"feat_{feature}"


        # 创建带第二 y 轴的图表

        fig = make_subplots(specs=[[{"secondary_y": True}]])


        # 绘制真实值

        fig.add_trace(

            go.Scatter(x=output["timestamp"], y=output[p_col], name="True"),

            secondary_y=False,

        )


        fig.add_trace(

            go.Scatter(x=output["timestamp"], y=output[f'{p_col}|Pred'], name="Predicted"),

            secondary_y=False,

        )


        if plot_errors:

            fig.add_trace(

                go.Scatter(x=output["timestamp"], y=output[f"A_Score_{feature}"], name="Anomaly Score"),

                secondary_y=True,

            )


        if plot_feature_anom:

            fig.add_trace(

                go.Scatter(x=output["timestamp"], y=output[f"A_Pred_{feature}"], mode="markers", name="Predicted Anomaly"),

                secondary_y=False,

            )


            if self.labels_available:

                fig.add_trace(

                    go.Scatter(x=output["timestamp"], y=output[f"A_True_{feature}"], mode="markers", name="True Anomaly"),

                    secondary_y=False,

                )


        # 更新布局

        fig.update_layout(title_text=f"{title_prefix} Feature {feature} ({p_col})")


        # 更新 x 轴标题

        fig.update_xaxes(title_text="Timestamp")


        # 更新 y 轴标题

        fig.update_yaxes(title_text="Value", secondary_y=False)

        fig.update_yaxes(title_text="Score", secondary_y=True)


        # 保存图表

        save_path = f"{self.result_path}/{title_prefix.lower()}_feature_{feature}.html"

        py.offline.plot(fig, filename=save_path, auto_open=False)

        print(f"{title_prefix} feature {feature} plot saved to {save_path}")


    def plot_all_features(self, start=None, end=None, type="test"):

        """

        按以下顺序绘制所有特征：

            - forecasting for feature i

            - reconstruction for feature i

            - true value for feature i

            - anomaly score (error) for feature i

        """

        if type == "train":

            data_copy = self.train_output.copy()

        elif type == "test":

            data_copy = self.test_output.copy()


        data_copy = data_copy.drop(columns=['timestamp', 'A_Score_Global', 'Thresh_Global'])

        cols = [c for c in data_copy.columns if not (c.startswith('Thresh_') or c.startswith('A_Pred_'))]

        data_copy = data_copy[cols]


        if start is not None and end is not None:

            assert start < end

        if start is not None:

            data_copy = data_copy.iloc[start:, :]

        if end is not None:

            start = 0 if start is None else start

            data_copy = data_copy.iloc[: end - start, :]


        num_cols = data_copy.shape[1]

        plt.tight_layout()

        colors = ["gray", "gray", "gray", "r"] * (num_cols // 4) + ["b", "g"]

        data_copy.plot(subplots=True, figsize=(20, num_cols), ylim=(0, 1.5), style=colors)

        plt.show()


    def plot_anomaly_segments(self, type="test", num_aligned_segments=None, show_boring_series=False):

        """

        查找并可视化集体异常，即同一时间发生的特征级异常。

        """

        is_test = True

        if type == "train":

            data_copy = self.train_output.copy()

            is_test = False

        elif type == "test":

            data_copy = self.test_output.copy()


        def get_pred_cols(df):

            pred_cols_to_remove = []

            col_names_to_remove = []

            for i, col in enumerate(self.pred_cols):

                y = df[f"True_{i}"].values

                if np.average(y) >= 0.95 or np.average(y) == 0.0:

                    pred_cols_to_remove.append(col)

                    cols = list(df.columns[4 * i : 4 * i + 4])

                    col_names_to_remove.extend(cols)


            df.drop(col_names_to_remove, axis=1, inplace=True)

            return [x for x in self.pred_cols if x not in pred_cols_to_remove]


        non_constant_pred_cols = self.pred_cols if show_boring_series else get_pred_cols(data_copy)


        fig = make_subplots(

            rows=len(non_constant_pred_cols),

            cols=1,

            vertical_spacing=0.4 / len(non_constant_pred_cols),

            shared_xaxes=True,

        )


        timestamps = None

        shapes = []

        annotations = []

        for i in range(len(non_constant_pred_cols)):

            new_idx = int(data_copy.columns[4 * i].split("_")[-1])

            values = data_copy[f"True_{new_idx}"].values


            anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_Pred_{new_idx}"].values)


            y_min = -0.1

            y_max = 2  # 0.5 * y_max


            j = i + 1

            xref = f"x{j}" if i > 0 else "x"

            yref = f"y{j}" if i > 0 else "y"

            anomaly_shape = self.create_shapes(

                anomaly_sequences, None, y_min, y_max, None, xref=xref, yref=yref, is_test=is_test

            )

            shapes.extend(anomaly_shape)


            fig.append_trace(

                go.Scatter(x=timestamps, y=values, line=dict(color=get_series_color(values), width=1)), row=i + 1, col=1

            )

            fig.update_yaxes(range=[-0.1, get_y_height(values)], row=i + 1, col=1)


            annotations.append(

                dict(

                    # xref="paper",

                    xanchor="left",

                    yref=yref,

                    text=f"<b>{non_constant_pred_cols[i].upper()}</b>",

                    font=dict(size=10),

                    showarrow=False,

                    yshift=35,

                    xshift=(-523),

                )

            )


        colors = ["blue", "green", "red", "black", "orange", "brown", "aqua", "hotpink"]

        taken_shapes_i = []

        keep_segments_i = []

        corr_segments_count = 0

        for nr, i in enumerate(range(len(shapes))):

            corr_shapes = [i]

            shape = shapes[i]

            shape["opacity"] = 0.3

            shape_x = shape["x0"]


            for j in range(i + 1, len(shapes)):

                if j not in taken_shapes_i and shapes[j]["x0"] == shape_x:

                    corr_shapes.append(j)


            if num_aligned_segments is not None:

                if num_aligned_segments[0] == ">":

                    num = int(num_aligned_segments[1:])

                    keep_segment = len(corr_shapes) >= num

                else:

                    num = int(num_aligned_segments)

                    keep_segment = len(corr_shapes) == num


                if keep_segment:

                    keep_segments_i.extend(corr_shapes)

                    taken_shapes_i.extend(corr_shapes)

                    if len(corr_shapes) != 1:

                        for shape_i in corr_shapes:

                            shapes[shape_i]["fillcolor"] = colors[corr_segments_count % len(colors)]

                        corr_segments_count += 1


        if num_aligned_segments is not None:

            shapes = np.array(shapes)

            shapes = shapes[keep_segments_i].tolist()


        fig.update_layout(

            height=1800,

            width=1200,

            shapes=shapes,

            template="simple_white",

            annotations=annotations,

            showlegend=False)


        fig.update_yaxes(ticks="", showticklabels=False, showline=True, mirror=True)

        fig.update_xaxes(ticks="", showticklabels=False, showline=True, mirror=True)

        py.offline.iplot(fig)


    def plotly_global_predictions(self, type="test"):

        is_test = True

        if type == "train":

            data_copy = self.train_output.copy()

            is_test = False

        elif type == "test":

            data_copy = self.test_output.copy()


        tot_anomaly_scores = data_copy["A_Score_Global"].values

        pred_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_Pred_Global"].values)

        threshold = data_copy['Thresh_Global'].values

        y_min = -0.1

        y_max = 5 * np.mean(threshold) # np.max(tot_anomaly_scores)

        shapes = self.create_shapes(pred_anomaly_sequences, "pred", y_min, y_max, None, is_test=is_test)

        if self.labels_available and is_test:

            true_anomaly_sequences = self.get_anomaly_sequences(data_copy[f"A_True_Global"].values)

            shapes2 = self.create_shapes(true_anomaly_sequences, "true", y_min, y_max, None, is_test=is_test)

            shapes.extend(shapes2)


        layout = {

            "title": f"{type} set | Total error, predicted anomalies in blue, true anomalies in red if available "

                     f"(making correctly predicted in purple)",

            "shapes": shapes,

            "yaxis": dict(range=[0, y_max]),

            "height": 400,

            "width": 1500

        }


        fig = go.Figure(

            data=[go.Scatter(x=data_copy["timestamp"], y=tot_anomaly_scores, name='Error', line=dict(width=1, color="red")),

                  go.Scatter(x=data_copy["timestamp"], y=threshold, name='Threshold', line=dict(color="black", width=1, dash="dash"))],

            layout=layout,

        )

        py.offline.iplot(fig)


    def plot_attention_map(self, adj_matrix, title="Attention Map"):

        """

        可视化注意力图或连接矩阵

        :param adj_matrix: Tensor shape (batch, K, K)

        """

        adj_matrix = adj_matrix[0].cpu().detach().numpy()  # 取第一个样本

        plt.figure(figsize=(10, 8))

        sns.heatmap(adj_matrix, cmap='viridis', annot=False)

        plt.title(title)

        plt.show()


