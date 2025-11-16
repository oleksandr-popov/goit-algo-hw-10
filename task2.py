import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


def f(x):
    """The function to be integrated: f(x) = x^2 + 4x - 3"""
    return x**2 + x * 4 - 3


LOWER_BOUND = 0  # Lower limit of integration (a)
UPPER_BOUND = 5  # Upper limit of integration (b)


## Monte Carlo Integration Function
def mc_integral(a, b, num_samples=10000):
    """
    Estimates the definite integral using the Monte Carlo method (sampling).
    It works by finding the ratio of random points under the curve to the total points
    within a defined bounding box.
    """
    max_y = f(b)

    x_rand = np.random.uniform(a, b, num_samples)
    y_rand = np.random.uniform(0, max_y, num_samples)

    # Count how many random points (x, y) satisfy y <= f(x)
    points_under_curve = np.sum(y_rand <= f(x_rand))

    # Calculate the ratio of points under the curve to total points
    area_ratio = points_under_curve / num_samples

    # Estimate the integral (Area under curve)
    total_area_box = (b - a) * max_y

    integral_estimate = total_area_box * area_ratio

    return integral_estimate


if __name__ == "__main__":
    a = LOWER_BOUND
    b = UPPER_BOUND

    # Calculate the highly accurate numerical integral using SciPy's quad
    numerical_integral, numerical_error = sci.quad(f, a, b)
    print(
        f"**SciPy Numerical Integral (Reference):** {numerical_integral:.6f} (Error: {numerical_error:.2e})"
    )

    # Test Monte Carlo integral with varying sample sizes
    print("\n**Monte Carlo Estimates:**")
    monte_carlo_samples = [100, 1000, 10000, 100000, 1000000]

    for sample in monte_carlo_samples:
        mc_result = mc_integral(a, b, num_samples=sample)
        print(f"  MC Integral ({sample:>7,} samples): {mc_result:.6f}")

    # Plot the function and the integrated area
    x_plot = np.linspace(a - 0.5, b + 0.5, 400)
    y_plot = f(x_plot)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the function curve
    ax.plot(x_plot, y_plot, "r", linewidth=2, label="$f(x) = x^2 + 4x - 3$")

    # Fill the area under the curve between a and b
    ix_fill = np.linspace(a, b)
    iy_fill = f(ix_fill)
    ax.fill_between(ix_fill, iy_fill, color="gray", alpha=0.4, label="Integrated Area")

    # Set plot limits and labels
    ax.set_xlim([x_plot[0], x_plot[-1]])
    ax.set_ylim([-5, max(y_plot) + 5])  # Adjusted y-limit for better visualization
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    # Add vertical lines for the integration bounds
    ax.axvline(x=a, color="green", linestyle="--", label=f"Lower Bound (a={a})")
    ax.axvline(x=b, color="blue", linestyle="--", label=f"Upper Bound (b={b})")

    ax.set_title(
        f"Graphical Integration of $f(x) = x^2 + 4x - 3$ from $x={a}$ to $x={b}$"
    )
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
