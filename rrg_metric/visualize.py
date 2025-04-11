import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Optional, Dict, Any, Tuple
from scipy import stats

def plot_corr(
    metric: str,
    metric_scores: List[float],
    radiologist_error_counts: List[float],
    error_type: str = "total",
    ax: Optional[plt.Axes] = None,
    **params: Dict[str, Any]
) -> Tuple[plt.Axes, float, Tuple[float, float]]:
    """
    Create a correlation plot between metric scores and radiologist error counts.
    
    Args:
        metric (str): Name of the metric
        metric_scores (List[float]): List of metric scores
        radiologist_error_counts (List[float]): List of radiologist error counts
        error_type (str, optional): Type of error to plot. Must be either "total" or "significant". 
            Defaults to "total".
        ax (Optional[plt.Axes]): Matplotlib axes to plot on. If None, creates new figure and axes.
        **params: Additional parameters for customizing the plot
            - color (str): Color for points and regression line (default: 'blue')
            - width (int): Width of the figure in inches (default: 6)
            - height (int): Height of the figure in inches (default: 4)
            - xlabel (str): Label for x-axis (default: depends on error_type)
            - ylabel (str): Label for y-axis (default: "1 - Score")
            - alpha (float): Transparency of confidence interval (default: 0.2)
            - scatter_alpha (float): Transparency of scatter points (default: 0.6)
            - show_tau (bool): Whether to show Kendall's tau (default: True)
    
    Returns:
        Tuple[plt.Axes, float, Tuple[float, float]]: The matplotlib axes, tau, and the 95% confidence interval of tau
    
    Raises:
        ValueError: If error_type is not "total" or "significant"
    """
    metric_scores = 1 - np.array(metric_scores)
    radiologist_error_counts = np.array(radiologist_error_counts)

    if error_type not in ["total", "significant"]:
        raise ValueError('error_type must be either "total" or "significant"')

    # Default parameters
    default_params = {
        'color': 'blue',
        'width': 6,
        'height': 4,
        'xlabel': f'{error_type.capitalize()} Errors',
        'ylabel': '1 - Score',
        'alpha': 0.2,
        'scatter_alpha': 0.6,
        'show_tau': True
    }
    plot_params = {**default_params, **params}

    # Calculate correlation and confidence interval
    tau, _ = stats.kendalltau(radiologist_error_counts, metric_scores)

    # Bootstrap for confidence interval
    n_samples = len(metric_scores)
    bootstrap_taus = []
    for _ in range(1000):
        indices = np.random.randint(0, n_samples, n_samples)  # Sample indices once
        boot_tau, _ = stats.kendalltau(
            radiologist_error_counts[indices],
            metric_scores[indices]
        )
        bootstrap_taus.append(boot_tau)

    ci_lower, ci_upper = np.percentile(bootstrap_taus, [2.5, 97.5])

    # Create or use provided axis
    if ax is None:
        _, ax = plt.subplots(figsize=(plot_params['width'], plot_params['height']))
    
    # Create scatter plot with regression line
    sns.regplot(
        x=radiologist_error_counts,
        y=metric_scores,
        color=plot_params['color'],
        scatter_kws={'alpha': plot_params['scatter_alpha']},
        line_kws={'color': plot_params['color']},
        ci=95,
        n_boot=1000,
        ax=ax
    )
    
    # Customize plot
    ax.set_xlabel(plot_params['xlabel'])
    ax.set_ylabel(plot_params['ylabel'])
    ax.set_title(f"{metric} (τ = {tau:.3f})" if plot_params['show_tau'] else metric)
    
    # Set confidence interval alpha
    if len(ax.collections) > 0:
        ax.collections[0].set_alpha(plot_params['alpha'])
    
    # Only apply tight_layout if we created the figure
    if ax.figure.get_axes()[0] == ax:
        plt.tight_layout()
    
    return ax, tau, (ci_lower, ci_upper)