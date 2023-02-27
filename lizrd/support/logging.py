from typing import Optional
import plotly

from abc import ABC, abstractmethod

from lizrd.core.misc import generate_random_string
import numpy as np
import os
import math

_CURRENT_LOGGER = None


def get_current_logger() -> Optional["AbstractLogger"]:
    return _CURRENT_LOGGER


class AbstractLogger(ABC):
    def __init__(self, logger, auxiliary_params=None):
        global _CURRENT_LOGGER
        self.instance_logger = logger
        _CURRENT_LOGGER = self
        self.auxiliary_params = auxiliary_params

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
        if isinstance(figure.data[0], plotly.graph_objs.Scattergl):
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

    def get_auxiliary_metrics(self, title: str, value: float, iteration: int):
        auxiliary_metrics = {}
        if self.auxiliary_params is None:
            return auxiliary_metrics

        if (
            "x_flops_scale" in self.auxiliary_params
            and self.auxiliary_params["x_flops_scale"]
        ):
            assert (
                self.auxiliary_params["model_size"] is not None
            ), "if using x_compute_scale, you must provide model_size"
            assert (
                self.auxiliary_params["batch_size"] is not None
            ), "if using x_compute_scale, you must provide batch_size"
            auxiliary_metrics[f"{title}_flops"] = {
                "value": value,
                "iteration": iteration
                * self.auxiliary_params["model_size"]
                * self.auxiliary_params["batch_size"],
            }

        if (
            "x_log_scale" in self.auxiliary_params
            and self.auxiliary_params["x_log_scale"]
        ):

            def new_metric_with_log_scale(metric):
                return {
                    "value": metric["value"],
                    "iteration": math.log(metric["iteration"]),
                }

            for metric in auxiliary_metrics:
                auxiliary_metrics[f"{metric}_log"] = new_metric_with_log_scale(metric)
            auxiliary_metrics[f"{title}_log"] = new_metric_with_log_scale(
                {"value": value, "iteration": iteration}
            )
            return auxiliary_metrics


class ClearMLLogger(AbstractLogger):
    pass


class NeptuneLogger(AbstractLogger):
    _TMP_PLOTS_DIR: str = "./tmp_plots"

    def __init__(self, logger):
        super().__init__(logger)
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
