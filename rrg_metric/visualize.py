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
    **params: Dict[str, Any]
) -> Tuple[plt.Figure, Tuple[float, float]]:
    """
    Create a correlation plot between metric scores and radiologist error counts.
    
    Args:
        metric_scores (List[float]): List of metric scores
        radiologist_error_counts (List[float]): List of radiologist error counts
        error_type (str, optional): Type of error to plot. Must be either "total" or "significant". 
            Defaults to "total".
        **params: Additional parameters for customizing the plot
            - color (str): Color for points and regression line (default: 'blue')
            - width (int): Width of the figure in inches (default: 8)
            - height (int): Height of the figure in inches (default: 6)
            - xlabel (str): Label for x-axis (default: depends on error_type)
            - ylabel (str): Label for y-axis (default: "1 - Score")
            - title (str): Title of the plot (default: None)
            - alpha (float): Transparency of confidence interval (default: 0.2)
            - scatter_alpha (float): Transparency of scatter points (default: 0.6)
            - show_tau (bool): Whether to show correlation formula (default: True)
    
    Returns:
        Tuple[plt.Figure, Tuple[float, float]]: The created matplotlib figure and the 95% confidence interval of tau
    
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
        'width': 8,
        'height': 6,
        'xlabel': f'{error_type.capitalize()} Errors',
        'ylabel': '1 - Score',
        'alpha': 0.2,  # transparency for 95% confidence interval
        'scatter_alpha': 0.6,
        'show_tau': True
    }
    
    # Update defaults with provided parameters
    plot_params = {**default_params, **params}
    
    # Calculate correlation and confidence interval
    tau, p_value = stats.kendalltau(radiologist_error_counts, metric_scores)
    
    # Calculate 95% CI for tau using bootstrap
    n_bootstrap = 1000
    bootstrap_taus = []
    n_samples = len(metric_scores)
    
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        boot_tau, _ = stats.kendalltau(
            radiologist_error_counts[indices],
            metric_scores[indices]
        )
        bootstrap_taus.append(boot_tau)
    
    ci_lower, ci_upper = np.percentile(bootstrap_taus, [2.5, 97.5])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(plot_params['width'], plot_params['height']))
    
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
    
    title = metric
    title += f' (τ = {tau:.3f})' if plot_params['show_tau'] else ''

    ax.set_title(title)
    
    # Set confidence interval alpha
    if len(ax.collections) > 0:  # Check if confidence interval exists
        ax.collections[0].set_alpha(plot_params['alpha'])
    
    # Tight layout
    plt.tight_layout()
    
    return fig, (ci_lower, ci_upper)