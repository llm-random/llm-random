import math
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import plotly

from lizrd.core.misc import generate_random_string

_CURRENT_LOGGER = None


def set_current_logger(logger: "AbstractLogger"):
    global _CURRENT_LOGGER
    _CURRENT_LOGGER = logger


def get_current_logger() -> Optional["AbstractLogger"]:
    return _CURRENT_LOGGER


class AbstractLogger(ABC):
    def __init__(self, logger, auxiliary_params=None):
        self.instance_logger = logger
        self.auxiliary_params = auxiliary_params
        set_current_logger(self)

    @abstractmethod
    def flush_if_necessary(self):
        pass

    @abstractmethod
    def report_scalar(
        self, *, title: str, value: float, iteration: int, series: Optional[str] = None
    ):
        raise NotImplementedError()

    @abstractmethod
    def report_plotly(
        self,
        *,
        figure: plotly.graph_objs.Figure,
        title,
        series,
        iteration,
    ):
        raise NotImplementedError()

    def potentially_log_plotly_figure_scalars(
        self,
        *,
        figure: plotly.graph_objs.Figure,
        title: str,
        series: Optional[str],
        iteration: int,
    ):
        if isinstance(figure.data[0], plotly.graph_objs.Scattergl) or isinstance(
            figure.data[0], plotly.graph_objs._scatter.Scatter
        ):
            x = figure.data[0].x
            y = figure.data[0].y
            pearson_correlation = np.corrcoef(x, y)[0, 1]
            if series is not None:
                series = f"{series} (pearson correlation)"
            else:
                series = "pearson correlation"
            self.report_scalar(
                title=title,
                series=series,
                value=pearson_correlation,
                iteration=iteration,
            )
        elif isinstance(figure.data[0], plotly.graph_objs.Histogram):
            mean = figure.data[0].x.mean()
            std = figure.data[0].x.std()
            if series is not None:
                series_mean = f"{series} (mean)"
                series_std = f"{series} (std)"
            else:
                series_mean = "mean"
                series_std = "std"
            self.report_scalar(
                title=title, series=series_mean, value=mean, iteration=iteration
            )
            self.report_scalar(
                title=title, series=series_std, value=std, iteration=iteration
            )
        else:
            print(
                f"Could not log scalars for plotly figure of type {type(figure.data[0])}"
            )

    @staticmethod
    def get_log_x_scale_metric(value: float, iteration: int):
        return {
            "value": value,
            "iteration": math.log(max(iteration, 1)),
        }

    def get_metric_with_flop_scale(self, value: float, iteration: int):
        assert (
            self.auxiliary_params["model_size"] is not None
        ), "if using x_compute_scale, you must provide model_size"
        assert (
            self.auxiliary_params["batch_size"] is not None
        ), "if using x_compute_scale, you must provide batch_size"

        return {
            "value": value,
            "iteration": iteration
            * self.auxiliary_params["model_size"]
            * self.auxiliary_params["batch_size"],
        }

    def get_auxiliary_metrics(self, title: str, value: float, iteration: int):
        if self.auxiliary_params is None or self.auxiliary_params == {}:
            return {}

        metric_x_flop = None
        auxiliary_metrics = {}

        if "x_flop" in self.auxiliary_params and self.auxiliary_params["x_flop"]:
            metric_x_flop = self.get_metric_with_flop_scale(value, iteration)
            auxiliary_metrics[f"{title}_(x_flop)"] = metric_x_flop

        if (
            "x_logarithmic" in self.auxiliary_params
            and self.auxiliary_params["x_logarithmic"]
        ):
            if metric_x_flop is not None:
                metric_x_flop_logarithmic = self.get_log_x_scale_metric(
                    metric_x_flop["value"], metric_x_flop["iteration"]
                )
                auxiliary_metrics[
                    f"{title}_(x_flop_logarithmic)"
                ] = metric_x_flop_logarithmic

            metric_logarithmic = self.get_log_x_scale_metric(value, iteration)
            auxiliary_metrics[f"{title}_(x_logarithmic)"] = metric_logarithmic

        return auxiliary_metrics


class ClearMLLogger(AbstractLogger):
    pass


class NeptuneLogger(AbstractLogger):
    _TMP_PLOTS_DIR: str = "./tmp_plots"

    def __init__(self, logger, auxiliary_params=None):
        super().__init__(logger, auxiliary_params)
        self.random_id = generate_random_string(8)
        os.makedirs(self._TMP_PLOTS_DIR, exist_ok=True)

    def _make_path(
        self, title: str, series: Optional[str] = None, iteration: Optional[int] = None
    ):
        parts = [title]
        if series is not None:
            parts.append(series)
        if iteration is not None:
            parts.append(str(iteration))
        return "/".join(parts)

    def _upload_with_tmp_file(self, path, obj, extension="html"):
        tmp_file = f"{self._TMP_PLOTS_DIR}/{generate_random_string(16)}.{extension}"
        with open(tmp_file, "w") as f:
            f.write(obj)
        self.instance_logger[path].upload(tmp_file)

    def report_scalar(
        self,
        *,
        title: str,
        value: float,
        iteration: int,
        series: Optional[str] = None,
    ):
        path = self._make_path(title, series, iteration)
        assert (not math.isnan(value)) and (
            not math.isinf(value)
        ), f"Trying to log {path} as {value}. Neptune doesn't allow logging NaN or Inf."
        self.instance_logger[self._make_path(title, series)].append(
            value=value, step=iteration
        )
        auxiliary_metrics = self.get_auxiliary_metrics(title, value, iteration)
        for metric_name, metric in auxiliary_metrics.items():
            self.instance_logger[self._make_path(metric_name, series)].append(
                value=metric["value"], step=metric["iteration"]
            )

    def report_plotly(
        self,
        *,
        figure: plotly.graph_objs.Figure,
        title: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        path = self._make_path(title, series, iteration)
        # log json
        json = figure.to_json()
        self._upload_with_tmp_file(f"{path}_json", json, "json")
        # log html
        html = figure.to_html(include_plotlyjs="cdn")
        self._upload_with_tmp_file(f"{path}_plot", html, "html")
        # log associated_scalars
        self.potentially_log_plotly_figure_scalars(
            figure=figure, title=title, series=series, iteration=iteration
        )

    def flush_if_necessary(self):
        pass


def log_plot(figure: plotly.graph_objs.Figure, title: str, series: str, iteration: int):
    logger = get_current_logger()
    assert logger is not None
    logger.report_plotly(figure=figure, title=title, series=series, iteration=iteration)
