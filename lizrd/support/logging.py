import plotly
from clearml import Logger


def log_plot_to_clearml(
    figure: plotly.graph_objs._figure.Figure, title: str, series: str, iteration: int
):
    """Utility function for loggin to plotly. Creates plots and scalars from plotly figures."""
    logger = Logger.current_logger()
    if isinstance(figure.data[0], plotly.graph_objs._scatter.Scatter):
        x = figure.data[0].x
        y = figure.data[0].y
        pearson_correlation = np.corrcoef(x, y)[0, 1]
        logger.report_scalar(
            title=title,
            series=series + " pearson correlation",
            value=pearson_correlation,
            iteration=iteration,
        )
        logger.report_plotly(
            title=title,
            series=series,
            figure=figure,
            iteration=iteration,
        )
    elif isinstance(figure.data[0], plotly.graph_objs._histogram.Histogram):
        mean = figure.data[0].x.mean()
        std = figure.data[0].x.std()
        logger.report_scalar(
            title=title, series=series + " mean", value=mean, iteration=iteration
        )
        logger.report_scalar(
            title=title, series=series + " std", value=std, iteration=iteration
        )
        logger.report_plotly(
            title=title,
            series=series,
            figure=figure,
            iteration=iteration,
        )
    else:
        logger.report_plotly(
            title=title,
            series=series,
            figure=figure,
            iteration=iteration,
        )
