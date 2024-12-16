from typing import List
from scipy import stats
import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import ceil
import phik
from sklearn.metrics import confusion_matrix, log_loss

COLOR_PALETTE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


def exclude_list_value(cols_list: list, value: str) -> list:
    """Returns a list of columns excluding the given value."""
    cols_list = [col for col in cols_list if col != value]

    return cols_list


def get_screen_width() -> int:
    """Retrieves the screen width using a tkinter root window and returns the screen width value."""
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    root.destroy()

    return screen_width


def set_font_size() -> dict:
    """Sets the font sizes for visualization elements based on the screen width."""
    base_font_size = round(get_screen_width() / 70, 0)
    font_sizes = {
        'font.size': base_font_size * 0.6,
        'axes.titlesize': base_font_size * 0.6,
        'axes.labelsize': base_font_size * 0.6,
        'xtick.labelsize': base_font_size * 0.4,
        'ytick.labelsize': base_font_size * 0.4,
        'legend.fontsize': base_font_size * 0.3,
        'figure.titlesize': base_font_size * 0.6,
    }

    return font_sizes


def train_test_missing_values(
        df_train: pd.DataFrame,
        df_test: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame that summarizes missing values in the train and test datasets."""
    missing_values_train = round(df_train.isnull().sum(), 0)
    missing_values_perc_train = round(
        (missing_values_train / len(df_train)) * 100, 1)
    missing_values_test = round(df_test.isnull().sum(), 0)
    missing_values_perc_test = round(
        (missing_values_test / len(df_test)) * 100, 1)

    missing_values = pd.DataFrame({
        'Train #': missing_values_train,
        'Train %': missing_values_perc_train,
        'Test #': missing_values_test,
        'Test %': missing_values_perc_test
    })

    return missing_values


def draw_numerical_distributions(
        df: pd.DataFrame,
        cols_list: list,
        grid_cols: int = 3,
        chart_type: str = 'histogram') -> None:
    """
    Draws histograms for the given DataFrame and columns.

    Args:
        df (DataFrame): Input DataFrame
        cols_list (list): List of columns to plot
        grid_cols (int): Number of columns in the grid
        chart_type (str): Type of chart to plot ('histogram', 'box', 'violin')

    Returns:
        None
    """
    # Grid size configuration
    cols = grid_cols
    rows = ceil(len(cols_list) / cols)
    fig_width = get_screen_width() / 100

    # Font configuration
    font_sizes = set_font_size()
    plt.rcParams.update(font_sizes)

    # Plotting
    _, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(
            fig_width, fig_width / cols))
    axes = axes.flatten()

    for i, col in enumerate(cols_list):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        if chart_type == 'histogram':
            sns.histplot(df[col], color=color, ax=axes[i])
        elif chart_type == 'box':
            sns.boxplot(df[col], color=color, ax=axes[i])
        elif chart_type == 'violin':
            sns.violinplot(df[col], color=color, ax=axes[i])
        elif chart_type == 'kde':
            sns.kdeplot(df[col], color=color, fill=True, ax=axes[i])
            mean = df[col].mean()
            axes[i].axvline(
                mean,
                color='r',
                linestyle='--',
                label=f'Mean: {
                    mean:.2f}')
            axes[i].legend()

        else:
            return ValueError('Invalid chart type')
        axes[i].grid(False)

    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def draw_categorical_distributions(
        df: pd.DataFrame,
        cols_list: list,
        grid_cols: int) -> None:
    """
    Draws bar plots for the given DataFrame and columns.

    Args:
        df (DataFrame): Input DataFrame
        cols_list (list): List of columns to plot
        grid_cols (int): Number of columns in the grid

    Returns:
        None
    """
    # Grid size configuration
    cols = grid_cols
    rows = ceil(len(cols_list) / cols)
    fig_width = get_screen_width() / 100

    # Font configuration
    font_sizes = set_font_size()
    plt.rcParams.update(font_sizes)

    # Plotting
    _, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(
            fig_width, fig_width / cols))
    axes = axes.flatten()

    for i, col in enumerate(cols_list):
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        if df[col].dtype == bool:
            df[col] = df[col].astype(str)
        sns.countplot(df[col], color=color, ax=axes[i])
        axes[i].grid(False)

    for j in range(i + 1, rows * cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def draw_predictor_numerical_plots(
        df: pd.DataFrame,
        predictor: str,
        target: str,
        hist_type='histogram') -> None:
    """
    Draws two plots to visualize the frequency counts and box plot of the distribution of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.
        hist_type (str): The type of plot to draw. Can be 'histogram' or 'kde'. Defaults to 'histogram'.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(fig_width, fig_width / 5))

    # Chart 1: Box Plot
    sns.boxplot(
        data=df,
        x=target,
        y=predictor,
        hue=target,
        palette=COLOR_PALETTE,
        saturation=0.75,  # Default saturation
        ax=ax1
    )
    ax1.set_title(f'Distribution of {predictor} by {target}')
    ax1.set_xlabel(f'{target}')
    ax1.set_ylabel(f'{predictor}')

    # Chart 2: Histogram
    if hist_type == 'kde':
        sns.kdeplot(
            data=df,
            x=predictor,
            hue=target,
            multiple='stack',
            palette=COLOR_PALETTE,
            ax=ax2)
    else:
        sns.histplot(
            data=df,
            x=predictor,
            hue=target,
            multiple='stack',
            palette=COLOR_PALETTE,
            ax=ax2)
    ax2.set_title(
        f'Frequency Distribution of {predictor.title()} by {target.title()}')
    ax2.set_xlabel(f'{predictor.title()}')
    ax2.set_ylabel('Count')

    plt.show()
    plt.close(fig)


def numerical_predictor_significance_test(
        df: pd.DataFrame,
        predictor: str,
        target: str,
        test_type='mann_whitney') -> dict:
    """
    Perform either Mann-Whitney U test or Mood's median test and return the p-value.

    Args:
    df (pandas.DataFrame): The dataframe containing the data
    predictor (str): The name of the column containing the numerical predictor
    target (str): The name of the column containing the binary target
    test_type (str): Either 'mann_whitney' or 'moods_median'. Default is 'mann_whitney'

    Returns:
    dict: A dictionary containing the test results
    """
    # Separate the data into two groups based on the binary target
    group1 = df[df[target] == 0][predictor]
    group2 = df[df[target] == 1][predictor]

    if test_type == 'mann_whitney':
        statistic, p_value = stats.mannwhitneyu(
            group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U test"
    elif test_type == 'moods_median':
        statistic, p_value, _, _ = stats.median_test(group1, group2)
        test_name = "Mood's median test"
    else:
        raise ValueError(
            "Invalid test_type. Choose either 'mann_whitney' or 'moods_median'")

    # Calculate effect size (Cliff's delta for Mann-Whitney, Cohen's d for
    # Mood's median)
    if test_type == 'mann_whitney':
        # Cliff's delta
        effect_size = 2 * statistic / (len(group1) * len(group2)) - 1
    else:
        # Cohen's d
        effect_size = (np.mean(group1) - np.mean(group2)) / \
            np.sqrt((np.std(group1, ddof=1) ** 2 + np.std(group2, ddof=1) ** 2) / 2)

    results = {
        'test_name': test_name,
        'p_value': p_value,
        'statistic': statistic,
        'effect_size': effect_size,
        'group1_median': np.median(group1),
        'group2_median': np.median(group2)
    }

    return results


def interpret_results_numerical(
        df: pd.DataFrame,
        results: dict,
        col_name: str) -> pd.DataFrame:
    """Interpret the results of the non-parametric test and store them in a DataFrame"""
    data = {
        'Column': col_name,
        'Test Name': [results['test_name']],
        'P-value': [round(results['p_value'], 6)],
        'Test Statistic': [round(results['statistic'], 2)],
        'Effect Size': [round(results['effect_size'], 4)],
        'Median Group 0': [results['group1_median']],
        'Median Group 1': [results['group2_median']],
        'Significance': ['Statistically significant' if results['p_value'] < 0.05 else 'Not statistically significant'],
        'Effect Magnitude': []
    }

    if results['test_name'] == "Mann-Whitney U test":
        if abs(results['effect_size']) < 0.2:
            effect_magnitude = "negligible"
        elif abs(results['effect_size']) < 0.5:
            effect_magnitude = "small"
        elif abs(results['effect_size']) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
    else:  # Mood's median test
        if abs(results['effect_size']) < 0.2:
            effect_magnitude = "small"
        elif abs(results['effect_size']) < 0.5:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"

    data['Effect Magnitude'].append(effect_magnitude)
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df


def draw_predictor_categorical_plots(
        df: pd.DataFrame,
        predictor: str,
        target: str) -> None:
    """
    Draws two plots to visualize the frequency counts and proportions of a predictor variable by a target variable.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        predictor (str): The name of the predictor variable.
        target (str): The name of the target variable.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(fig_width, fig_width / 5))

    # Ensure predictor and target is treated as categorical
    df[predictor] = df[predictor].astype('category')
    df[target] = df[target].astype('category')

    # Chart 1: Frequencies
    sns.countplot(
        data=df,
        x=predictor,
        hue=target,
        palette=COLOR_PALETTE,
        ax=ax1
    )
    ax1.set_title(f'Frequency Counts of {predictor} by {target}')
    ax1.set_xlabel(f'{predictor}')
    ax1.set_ylabel('Count')

    # Chart 2: Proportions
    props = df.groupby(predictor)[target].value_counts(normalize=True).unstack().reset_index().melt(id_vars=predictor)
    sns.barplot(
        data=props,
        x=predictor,
        y='value',
        hue=target,
        palette=COLOR_PALETTE,
        ax=ax2
    )
    ax2.set_title(f'Proportion of {target} by {predictor}')
    ax2.set_xlabel(f'{predictor}')
    ax2.set_ylabel('Proportion')
    ax2.legend().set_visible(False)
    ax2.tick_params(axis='x', rotation=0)

    plt.show()
    plt.close(fig)


def categorical_predictor_significance_test(
        df: pd.DataFrame,
        predictor: str,
        target: str) -> dict:
    """
    Performs chi-squared test for independence between a categorical predictor and binary target.

    Args:
    df (pandas.DataFrame): The dataframe containing the data
    predictor (str): The name of the column containing the categorical predictor
    target (str): The name of the column containing the binary target

    Returns:
    dict: A dictionary containing the test results
    """
    # Create a contingency table
    df[predictor] = df[predictor].astype('category')
    df[target] = df[target].astype('category')
    contingency_table = pd.crosstab(df[predictor], df[target])

    # Perform chi-squared test
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency_table)

    # Calculate Cramer's V for effect size
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))

    results = {
        'test_name': "Chi-squared test",
        'p_value': p_value,
        'chi2_statistic': chi2,
        'degrees_of_freedom': dof,
        'effect_size': cramer_v,
        'contingency_table': contingency_table
    }

    return results


def interpret_results_categorical(
        df: pd.DataFrame,
        results: dict,
        col_name: str) -> pd.DataFrame:
    """Interpret the results of the chi-squared test. Store the summary in a DataFrame"""
    data = {
        'Column': col_name,
        'Test Name': [results['test_name']],
        'P-value': [round(results['p_value'], 6)],
        'Chi-squared statistic': [round(results['chi2_statistic'], 2)],
        'Degrees of freedom': [round(results['degrees_of_freedom'], 4)],
        'Effect size (Cramer\'s V)': [round(results['effect_size'], 4)],
        'Significance': ['Statistically significant' if results['p_value'] < 0.05 else 'Not statistically significant'],
        'Effect Magnitude': []
    }

    # Interpret effect size (Cramer's V)
    if results['effect_size'] < 0.1:
        effect_magnitude = "negligible"
    elif results['effect_size'] < 0.3:
        effect_magnitude = "small"
    elif results['effect_size'] < 0.5:
        effect_magnitude = "medium"
    else:
        effect_magnitude = "large"
    data['Effect Magnitude'] = effect_magnitude

    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    return df


def phik_matrix(df: pd.DataFrame, numerical_columns: list) -> None:
    """
    Calculates the Phi_k correlation coefficient matrix for the given DataFrame and columns.

    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_columns (list): List of numerical columns.

    Returns:
        None
    """
    # Interval columns
    corr_matrix = df.phik_matrix(interval_cols=numerical_columns)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(get_screen_width() / 100 * 0.8, 8))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5)
    plt.title('Phi_k Correlation Heatmap')

    plt.show()


def draw_original_log_distribution(df: pd.DataFrame, feature: str) -> None:
    """
    Draws two plots to visualize the distributions of a feature variable and the log-transformed distribution.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature (str): The name of the feature to visualize.

    Returns:
        None
    """

    fig_width = get_screen_width() / 100
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
                                   figsize=(fig_width, fig_width / 5))

    # Chart 1: Original distribution
    df[feature].hist(color=COLOR_PALETTE[0], grid=False, ax=ax1)
    ax1.set_title(f'Original distribution of {feature}')
    ax1.set_xlabel(f'{feature}')
    ax1.set_ylabel('Count')

    # Chart 2: Log-transformed distribution
    np.log1p(df[feature]).hist(color=COLOR_PALETTE[1], grid=False, ax=ax2)
    ax2.set_title(f'Log Distribution of {feature}')
    ax2.set_xlabel(f'{feature}')
    ax2.set_ylabel('Count')

    plt.show()
    plt.close(fig)


def calculate_skewness(
        df: pd.DataFrame,
        feature: str,
        output: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and prints the skewness of a feature variable in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        feature (str): The name of the feature variable.
        output (pd.DataFrame): The output DataFrame to store the results.

    Returns:
        pd.DataFrame: The updated output DataFrame with the skewness result.
    """
    results = {
        'Feature': feature,
        'Skewness': df[feature].skew(),
        'Log transform': 'Yes' if abs(df[feature].skew()) > 0.5 else 'No'
    }
    output = pd.concat([output, pd.DataFrame(
        results, index=[0])], ignore_index=True)

    return output


def draw_confusion_matrix(y: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots the confusion matrix for the given true labels and predicted labels.

    Args:
        y (np.ndarray): Array of true labels.
        y_pred (np.ndarray): Array of predicted labels.

    Returns:
        None
    """
    fig_width = get_screen_width() / 100
    plt.figure(figsize=(fig_width / 2, fig_width / 3))
    cm = confusion_matrix(y, y_pred)
    cm_normalized = cm.astype('float') / cm.sum()
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    plt.show()



def weighted_average_ensemble(
        weights: List[float],
        *predictions: np.ndarray) -> np.ndarray:
    """
    Calculate the weighted average of multiple predictions.

    Parameters:
        weights (List[float]): A list of weights corresponding to each prediction.
        *predictions (List[np.ndarray]): A list of predictions to be averaged.

    Returns:
        np.ndarray: The weighted average of the predictions.
    """
    return np.average(np.array(predictions), axis=0, weights=weights)


def log_loss_func(weights: List[float], *args: np.ndarray) -> float:
    """
    Calculate the log loss of a set of predictions.

    Parameters:
        weights (List[float]): A list of weights corresponding to each prediction.
        *args: A variable number of arguments, where the first argument is the true labels (yt) and the remaining arguments are predictions.

    Returns:
        float: The log loss of the weighted average of the predictions.
    """
    yt = args[0]
    predictions = args[1:]
    final_prediction = weighted_average_ensemble(weights, *predictions)
    return log_loss(yt, final_prediction)
