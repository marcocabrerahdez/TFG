"""Plot functions."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def plot_avg_time(y_test: pd.DataFrame, y_pred: pd.DataFrame, model: str, type_train: str, name: str) -> None:
    """Plot the average time of the model.

    Args:
        y_test (pd.DataFrame): DataFrame with the test data.
        y_pred (pd.DataFrame): DataFrame with the predicted data.
        model (str): Name of the model.
        type_train (str): Type of training.
        name (str): Name of the model.

    Returns
        None
    """
    print('Plotting...')

    # Filter the columns
    y_test_l95ci_df = y_test.filter(regex='^(?!(AVG|U95CI)).*')
    y_test_u95ci_df = y_test.filter(regex='^(?!(AVG|L95CI)).*')
    y_test_df = y_test.filter(regex='^(?!(L95CI|U95CI)).*')

    y_pred_l95ci_df = y_pred.filter(regex='^(?!(AVG|U95CI)).*')
    y_pred_u95ci_df = y_pred.filter(regex='^(?!(AVG|L95CI)).*')
    y_pred_df = y_pred.filter(regex='^(?!(L95CI|U95CI)).*')

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 10))

    # Normalize the data
    ax.set_xlim([0,
                max(y_test_df.values.max(),
                    y_pred_df.values.max(),
                    y_test_l95ci_df.values.max(),
                    y_pred_l95ci_df.values.max(),
                    y_test_u95ci_df.values.max(),
                    y_pred_u95ci_df.values.max()) + 1])
    ax.set_ylim([0,
                max(y_test_df.values.max(),
                    y_pred_df.values.max(),
                    y_test_l95ci_df.values.max(),
                    y_pred_l95ci_df.values.max(),
                    y_test_u95ci_df.values.max(),
                    y_pred_u95ci_df.values.max()) + 1])

    # Plot the data
    ax.plot([0,
            max(y_test_df.values.max(),
                y_pred_df.values.max(),
                y_test_l95ci_df.values.max(),
                y_pred_l95ci_df.values.max(),
                y_test_u95ci_df.values.max(),
                y_pred_u95ci_df.values.max()) + 1],
            [0,
            max(y_test_df.values.max(),
                y_pred_df.values.max(),
                y_test_l95ci_df.values.max(),
                y_pred_l95ci_df.values.max(),
                y_test_u95ci_df.values.max(),
                y_pred_u95ci_df.values.max()) + 1],
            color='black',
            linestyle='-',
            label='Valor Ideal')

    # Create the convex hull of the data
    avg_points = np.column_stack((y_test_df, y_pred_df))
    avg_hull = ConvexHull(avg_points)

    # Plot the convex hull
    ax.fill(avg_points[avg_hull.vertices, 0],
            avg_points[avg_hull.vertices, 1],
            'r',
            alpha=0.15,
            label='Área de valores predichos de tiempo promedio')
    ax.scatter(y_test_df, y_pred_df, color='red', alpha=0.85,
               label='Valor predicho de tiempo promedio')

    # Plot the L95CI
    l95ci_points = np.column_stack(
        (y_test_l95ci_df, y_pred_l95ci_df))
    l95ci_hull = ConvexHull(l95ci_points)

    ax.fill(l95ci_points[l95ci_hull.vertices, 0],
            l95ci_points[l95ci_hull.vertices, 1],
            color='blue',
            alpha=0.15,
            label='Área de valores predichos del intervalo de confianza inferior')
    ax.scatter(
        y_test_l95ci_df,
        y_pred_l95ci_df,
        color='blue',
        marker='x',
        alpha=0.85,
        label='Valor predicho del intervalo de confianza inferior')

    # Plot the U95CI
    u95ci_points = np.column_stack(
        (y_test_u95ci_df, y_pred_u95ci_df))
    u95ci_hull = ConvexHull(u95ci_points)

    ax.fill(u95ci_points[u95ci_hull.vertices, 0],
            u95ci_points[u95ci_hull.vertices, 1],
            color='green',
            alpha=0.15,
            label='Área de valores predichos de intervalo de confianza superior')
    ax.scatter(
        y_test_u95ci_df,
        y_pred_u95ci_df,
        color='green',
        marker='^',
        alpha=0.85,
        label='Valor predicho del intervalo de confianza superior')

    # Add the labels
    ax.set_xlabel('Valor real de tiempo promedio',
                  fontsize=10, fontweight='bold')
    ax.set_ylabel('Valor predicho de tiempo promedio',
                  fontsize=10, fontweight='bold')

    # Add the legend
    ax.legend()

    ax.set_title(
        f'Tiempo promedio hasta aparición de {name}',
        fontweight='bold',
        fontsize=11)

    # Configure the layout
    fig.set_layout_engine('compressed')

    # Save the figure
    if not os.path.exists(
        os.path.join(
            type_train,
            model)):
        os.makedirs(os.path.join(type_train,
                    model))
    plt.savefig(
        os.path.join(
            type_train,
            model,
            f'{name}.png'))

    # Close the figure
    plt.close()


def plot_upto_time(y_test: pd.DataFrame, y_pred: pd.DataFrame, model: str, type_train: str, name: str) -> None:
    """Plots the predicted values of the models up to the time of the test data.

    Args:
        y_test (pd.DataFrame): The test data.
        y_pred (pd.DataFrame): The predicted values of the models.
        model (str): The name of the model.
        type_train (str): The type of training.
        name (str): The name of the model.

    Returns:
        None
    """
    print('Plotting...')

    # Filter the data
    y_test_l95ci_df = y_test.filter(regex='^(?!(AVG|U95CI)).*')
    y_test_u95ci_df = y_test.filter(regex='^(?!(AVG|L95CI)).*')
    y_test_df = y_test.filter(regex='^(?!(L95CI|U95CI)).*')

    y_pred_l95ci_df = y_pred.filter(regex='^(?!(AVG|U95CI)).*')
    y_pred_u95ci_df = y_pred.filter(regex='^(?!(AVG|L95CI)).*')
    y_pred_df = y_pred.filter(regex='^(?!(L95CI|U95CI)).*')

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(y_test_df.columns, y_test_df.cumsum(
        axis=1).loc[1], 'o-b', label='Valor ideal')
    ax.plot(y_pred_df.columns, y_pred_df.cumsum(
        axis=1).loc[1], 'o-r', label='Valor predicho')

    # Plot the confidence intervals
    ax.fill_between(
        y_test_df.columns,
        y_test_l95ci_df.cumsum(
            axis=1).loc[1],
        y_test_u95ci_df.cumsum(
            axis=1).loc[1],
        alpha=0.2,
        color='b',
        label='Área de confianza real')
    ax.fill_between(
        y_pred_df.columns,
        y_pred_l95ci_df.cumsum(
            axis=1).loc[1],
        y_pred_u95ci_df.cumsum(
            axis=1).loc[1],
        alpha=0.2,
        color='r',
        label='Área de confianza predicha')

    # Add the ticks
    ax.set_xticks(y_test_df.columns)
    ax.set_xticklabels([i.split("UPTO_")[-1]
                        for i in y_test_df.columns])

    # Add the labels
    ax.set_xlabel('Edad')
    ax.set_ylabel('Porcentaje de afectación')

    ax.legend()

    # Add the title
    title = name.replace(' (UPTO)', '')
    fig.suptitle(
        f'Relación de afectación de {title} por grupos de edad',
        fontweight='bold',
        fontsize=12)

    # Configure the layout
    fig.set_layout_engine('compressed')

    if not os.path.exists(
        os.path.join(
            type_train,
            model)):
        os.makedirs(os.path.join(type_train,
                    model))
    plt.savefig(
        os.path.join(
            type_train,
            model,
            f'{name}.png'))

    # Close the plot
    plt.close()
