import math
import os
import secrets
from abc import ABC, abstractmethod
from argparse import Namespace
from typing import List, Optional

import neptune
import numpy as np
import plotly
import plotly.express as px
import torch
import wandb

from lizrd.support.misc import (
    make_concise_datetime,
    tags_to_name,
    count_parameters,
    generate_random_string,
)

_CURRENT_LOGGER = None


def set_current_logger(logger: "AbstractLogger"):
    global _CURRENT_LOGGER
    _CURRENT_LOGGER = logger


def get_current_logger() -> Optional["AbstractLogger"]:
    return _CURRENT_LOGGER


class AbstractLogger(ABC):
    def __init__(self, logger, args: Namespace):
        self.instance_logger = logger
        self.args = vars(args)

    @abstractmethod
    def report_scalar(
        self, *, title: str, value: float, iteration: int, series: Optional[str] = None
    ):
        raise NotImplementedError()

    @abstractmethod
    def report_text(
        self, *, title: str, value: str, iteration: int, series: Optional[str] = None
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

    @abstractmethod
    def report_generic_info(self, *, title: str, iteration: int, data):
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
            pass  # removed warning because it's too verbose and can't debug

    @staticmethod
    def get_log_x_scale_metric(value: float, iteration: int):
        return {
            "value": value,
            "iteration": math.log(max(iteration, 1)),
        }

    def get_metric_with_flop_scale(self, value: float, iteration: int):
        return {
            "value": value,
            "iteration": iteration
            * self.args["model_n_params"]
            * self.args["batch_size"],
        }

    def get_auxiliary_metrics(self, title: str, value: float, iteration: int):
        if not self.args.get("x_flop") and not self.args.get("log_x_scale"):
            return {}

        metric_x_flop = None
        auxiliary_metrics = {}

        if self.args.get("x_flop"):
            metric_x_flop = self.get_metric_with_flop_scale(value, iteration)
            auxiliary_metrics[f"{title}_(x_flop)"] = metric_x_flop

        if self.args.get("x_logarithmic"):
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

    def __init__(self, logger, args: Namespace):
        super().__init__(logger, args)
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

    def report_generic_info(self, *, title: str, iteration: int, data):
        if isinstance(data, plotly.graph_objs.Figure):
            self.report_plotly(figure=data, title=title, iteration=iteration)
        elif isinstance(data, list):
            if isinstance(data[0], float):
                for i, scalar in enumerate(data):
                    self.report_scalar(
                        title=title, value=scalar, series=str(i), iteration=iteration
                    )
            else:
                raise NotImplementedError()
        else:
            self.report_scalar(title=title, value=data, iteration=iteration)

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

    def report_text(
        self,
        *,
        title: str,
        value: float,
        iteration: int,
        series: Optional[str] = None,
    ):
        self.instance_logger[self._make_path(title, series)].append(
            value=value, step=iteration
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
        directory, filename = path.rsplit("/", 1)
        # log json
        json = figure.to_json()
        self._upload_with_tmp_file(f"{directory}/json_{filename}", json, "json")
        # log html
        html = figure.to_html(include_plotlyjs="cdn")
        self._upload_with_tmp_file(f"{directory}/plot_{filename}", html, "html")
        # log associated_scalars
        self.potentially_log_plotly_figure_scalars(
            figure=figure, title=title, series=series, iteration=iteration
        )


class WandbLogger(AbstractLogger):
    def __init__(self, logger, args: Namespace):
        super().__init__(logger, args)
        self.random_id = generate_random_string(8)

    def _make_path(self, title: str, series: Optional[str] = None):
        parts = [title]
        if series is not None:
            parts.append(series)
        return "/".join(parts)

    def report_generic_info(self, *, title: str, iteration: int, data):
        if isinstance(data, plotly.graph_objs.Figure):
            self.report_plotly(figure=data, title=title, iteration=iteration)
        elif isinstance(data, list):
            if isinstance(data[0], float):
                for i, scalar in enumerate(data):
                    self.report_scalar(
                        title=title, value=scalar, series=str(i), iteration=iteration
                    )
            else:
                raise NotImplementedError()
        else:
            self.report_scalar(title=title, value=data, iteration=iteration)

    def report_scalar(
        self,
        *,
        title: str,
        value: float,
        iteration: int,
        series: Optional[str] = None,
    ):
        path = self._make_path(title, series)
        wandb.log({path: value, "train/step": iteration})
        auxiliary_metrics = self.get_auxiliary_metrics(title, value, iteration)
        for metric_name, metric in auxiliary_metrics.items():
            wandb.log({metric_name: metric["value"], "train/step": iteration})

    def report_text(
        self,
        *,
        title: str,
        value: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        table = wandb.Table(columns=["value"], data=[[value]])
        wandb.log({self._make_path(title, series): table, "train/step": iteration})

    def report_plotly(
        self,
        *,
        figure: plotly.graph_objs.Figure,
        title: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        wandb.log({self._make_path(title, series): figure, "train/step": iteration})
        self.potentially_log_plotly_figure_scalars(
            figure=figure, title=title, series=series, iteration=iteration
        )
        return


class StdoutLogger(AbstractLogger):
    def print_out_metric(
        self,
        title: str,
        value: float,
        iteration: int,
        series: Optional[str] = None,
    ):
        ITERATION_SPACE = 7
        NAME_SPACE = 40
        info = f"/{series}" if series is not None else ""
        name = f"{title}{info}"
        space_1 = max(0, ITERATION_SPACE - len(str(iteration))) * " "
        space_2 = max(0, NAME_SPACE - len(name)) * " "
        print(f"Step:{iteration}{space_1}{name}{space_2} ==> {value} ")

    def report_generic_info(self, *, title: str, iteration: int, data):
        if isinstance(data, plotly.graph_objs.Figure):
            self.report_plotly(figure=data, title=title, iteration=iteration)
        elif isinstance(data, list):
            if isinstance(data[0], float):
                for i, scalar in enumerate(data):
                    self.report_scalar(
                        title=title, value=scalar, series=str(i), iteration=iteration
                    )
            else:
                raise NotImplementedError()
        else:
            self.report_scalar(title=title, value=data, iteration=iteration)

    def report_scalar(
        self,
        *,
        title: str,
        value: float,
        iteration: int,
        series: Optional[str] = None,
    ):
        self.print_out_metric(
            title=title, value=value, iteration=iteration, series=series
        )

    def report_text(
        self,
        *,
        title: str,
        value: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        self.print_out_metric(
            title=title, value=value, iteration=iteration, series=series
        )

    def report_plotly(
        self,
        *,
        figure: plotly.graph_objs.Figure,
        title: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        pass


class JointLogger(AbstractLogger):
    def __init__(self, loggers: List[AbstractLogger]):
        self.loggers = loggers
        set_current_logger(self)

    def report_generic_info(self, *, title: str, iteration: int, data):
        for logger in self.loggers:
            logger.report_generic_info(title=title, iteration=iteration, data=data)

    def report_scalar(
        self,
        *,
        title: str,
        value: float,
        iteration: int,
        series: Optional[str] = None,
    ):
        for logger in self.loggers:
            logger.report_scalar(
                title=title, value=value, iteration=iteration, series=series
            )

    def report_text(
        self,
        *,
        title: str,
        value: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        for logger in self.loggers:
            logger.report_text(
                title=title, value=value, iteration=iteration, series=series
            )

    def report_plotly(
        self,
        *,
        figure: plotly.graph_objs.Figure,
        title: str,
        iteration: int,
        series: Optional[str] = None,
    ):
        for logger in self.loggers:
            logger.report_plotly(
                figure=figure, title=title, iteration=iteration, series=series
            )


def log_plot(figure: plotly.graph_objs.Figure, title: str, series: str, iteration: int):
    logger = get_current_logger()
    assert logger is not None
    logger.report_plotly(figure=figure, title=title, series=series, iteration=iteration)


def get_logger(args, model, VOCAB_SIZE):
    timestamp = make_concise_datetime()
    unique_timestamp = f"{timestamp}{secrets.token_urlsafe(1)}"
    if args.logger_types == "":
        logger_types = []
    else:
        logger_types = args.logger_types.split(",")
        assert len(logger_types) == len(set(logger_types)), "Duplicate logger types."
    for arg in ["grad_modif_params", "grad_modif_placement"]:
        val = getattr(args, arg)
        if val is not None and isinstance(val, list):
            setattr(args, arg, ",".join(val))
    initialized_loggers = []
    for logger_type in logger_types:
        if logger_type == "neptune":
            run = neptune.init_run(
                project=args.project_name,
                tags=args.tags,
                name=f"{args.name} {tags_to_name(args.tags)} {unique_timestamp}",
            )
            run["args"] = vars(args)
            run["working_directory"] = os.getcwd()
            run["config"].upload(args.path_to_entry_config)
            all_config_paths = args.all_config_paths.split(",")
            run["all_configs"].upload_files(all_config_paths)

            args.model_n_params = count_parameters(model, args, VOCAB_SIZE)
            initialized_loggers.append(NeptuneLogger(run, args))
        elif logger_type == "wandb":
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=f"{args.name} {tags_to_name(args.tags)} {unique_timestamp}",
                tags=args.tags,
                config=vars(args),
            )
            # define our custom x axis metric
            wandb.define_metric("train/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("*", step_metric="train/step")
            initialized_loggers.append(WandbLogger(wandb, args))
        elif logger_type == "stdout":
            initialized_loggers.append(StdoutLogger(None, args))
        else:
            raise NotImplementedError(
                f"Logger of type '{logger_type}' is not implemented."
            )
    return JointLogger(initialized_loggers)


def prepare_tensor_for_logging(
    x: torch.Tensor, sample_size=2500, with_replacement=False
):
    """Prepare tensor or tensors for logging by sampling it to a maximum of `sample_size` elements.
    Default sample size = 2500 is selected because (experimentally) this works with ClearML plotting
    """
    num_elems = x.numel()
    x = x.detach().view(-1).cpu().numpy()

    if num_elems <= sample_size:
        return x.tolist()

    random_indices = np.random.choice(num_elems, sample_size, replace=with_replacement)
    ret_val = x[random_indices].tolist()
    return ret_val


def make_histogram(tensor, **kwargs):
    return px.histogram(
        prepare_tensor_for_logging(tensor, with_replacement=False), **kwargs
    )
