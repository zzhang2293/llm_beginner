"""Pictures for coffee_prediction.ipynb.

You do not need to read or change this file.
"""

import numpy as np

_TEMPERATURE_RANGE = (150.0, 285.0)  # degrees Celsius
_DURATION_RANGE = (11.5, 15.5)       # minutes

# The true rule makes a triangle of good coffees. These are its corners.
_GOOD_TRIANGLE = np.array([
    [175.0, 12.0],
    [255.0, 12.0],
    [175.0, 21.0 - 3.0 * 175.0 / 85.0],
    [175.0, 12.0],
])


def make_grid(n=100):
    """Makes n * n raw coffees that cover all temperatures and durations of the data.

    Returns:
        An array with shape (n * n, 2). Each row is [temperature, duration].
    """
    temperatures = np.linspace(*_TEMPERATURE_RANGE, n)
    durations = np.linspace(*_DURATION_RANGE, n)
    T, D = np.meshgrid(temperatures, durations)
    return np.column_stack([T.ravel(), D.ravel()])


def plot_data(X, Y):
    """Draws the coffees. The good coffees are inside the dashed triangle."""
    plt = _pyplot()
    if plt is None:
        return
    _, ax = plt.subplots(figsize=(6, 4.5))
    _draw_coffees(ax, X, Y)
    ax.plot(_GOOD_TRIANGLE[:, 0], _GOOD_TRIANGLE[:, 1], "k--", linewidth=1,
            label="true rule: good region")
    ax.set_title("The coffee data")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


def plot_network(X, Y, X_grid, A1_grid, A2_grid):
    """Draws the output of each neuron for each coffee of the grid.

    Args:
        X, Y: the training coffees and their labels.
        X_grid: the raw grid coffees from make_grid(), shape (n * n, 2).
        A1_grid: the output of layer 1 for each grid coffee, shape (n * n, 3).
        A2_grid: the output of layer 2 for each grid coffee, shape (n * n, 1).
    """
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.lines import Line2D  # pylint: disable=import-outside-toplevel

    X_grid = np.asarray(X_grid)
    A1_grid = np.asarray(A1_grid)
    A2_grid = np.asarray(A2_grid)
    n = int(round(np.sqrt(len(X_grid))))
    T = X_grid[:, 0].reshape(n, n)
    D = X_grid[:, 1].reshape(n, n)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4), layout="constrained")
    for j in range(3):
        image = _draw_map(axes[j], T, D, A1_grid[:, j].reshape(n, n), "Blues")
        _draw_coffees(axes[j], X, Y)
        axes[j].set_title(f"Layer 1, neuron {j + 1}: a1[{j}]")
    fig.colorbar(image, ax=axes[:3], shrink=0.9, label="neuron output")

    ax = axes[3]
    a2 = A2_grid[:, 0].reshape(n, n)
    image = _draw_map(ax, T, D, a2, "Greens")
    ax.contour(T, D, a2, levels=[0.5], colors="black", linewidths=1.5)
    ax.plot(_GOOD_TRIANGLE[:, 0], _GOOD_TRIANGLE[:, 1], "k--", linewidth=1)
    _draw_coffees(ax, X, Y)
    ax.set_title("Layer 2: a2 = P(good)")
    fig.colorbar(image, ax=ax, shrink=0.9, label="probability of good coffee")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markerfacecolor="gold",
               markeredgecolor="black", label="good coffee"),
        Line2D([], [], marker="x", linestyle="", color="red", label="bad coffee"),
        Line2D([], [], color="black", linewidth=1.5, label="network: a2 = 0.5"),
        Line2D([], [], color="black", linestyle="--", linewidth=1, label="true rule"),
    ], loc="upper right", fontsize=7)
    plt.show()


def _draw_map(ax, T, D, values, cmap):
    """Colors each grid point by its value from 0 to 1."""
    return ax.pcolormesh(T, D, values, cmap=cmap, vmin=0.0, vmax=1.0, shading="auto")


def _draw_coffees(ax, X, Y):
    """Draws good coffees as gold circles and bad coffees as red crosses."""
    good = np.asarray(Y)[:, 0] == 1
    ax.scatter(X[good, 0], X[good, 1], marker="o", s=24, facecolor="gold",
               edgecolor="black", linewidth=0.6, label="good coffee")
    ax.scatter(X[~good, 0], X[~good, 1], marker="x", s=18, color="red",
               linewidth=1, label="bad coffee")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Duration (min)")


def _pyplot():
    """Returns matplotlib.pyplot, or None if matplotlib is not installed."""
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("The picture needs matplotlib. To install it, run this in a new cell:")
        print("    %pip install matplotlib")
        print("Then restart the kernel and run the notebook again.")
        return None
    return plt
