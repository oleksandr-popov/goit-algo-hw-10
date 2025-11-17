import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


def f(x):
    """Функція для інтегрування: f(x) = x^2 + 4x - 3"""
    return x**2 + x * 4 - 3


LOWER_BOUND = 0  # Нижня межа інтегрування (a)
UPPER_BOUND = 5  # Верхня межа інтегрування (b)


## Функція інтегрування методом Монте-Карло
def mc_integral(a, b, num_samples=10000):
    """
    Оцінює визначений інтеграл за допомогою методу Монте-Карло (вибірка).
    Працює шляхом знаходження співвідношення випадкових точок під кривою до
    загальної кількості точок у визначеному обмежувальному прямокутнику.
    """
    max_y = f(b)

    # Генеруємо випадкові координати x рівномірно розподілені між a та b
    x_rand = np.random.uniform(a, b, num_samples)
    # Генеруємо випадкові координати y рівномірно розподілені між 0 та max_y
    y_rand = np.random.uniform(0, max_y, num_samples)

    # Рахуємо, скільки випадкових точок (x, y) задовольняють умову y <= f(x) (тобто знаходяться під кривою)
    points_under_curve = np.sum(y_rand <= f(x_rand))

    # Обчислюємо співвідношення точок під кривою до загальної кількості точок
    area_ratio = points_under_curve / num_samples

    # Оцінюємо інтеграл (Площа під кривою)
    # Площа обмежувального прямокутника: (b - a) * max_y
    total_area_box = (b - a) * max_y

    # Оцінка інтеграла = Площа прямокутника * Співвідношення площ
    integral_estimate = total_area_box * area_ratio

    return integral_estimate


if __name__ == "__main__":
    a = LOWER_BOUND
    b = UPPER_BOUND

    # Обчислюємо високоточний чисельний інтеграл за допомогою SciPy's quad
    numerical_integral, numerical_error = sci.quad(f, a, b)
    print(
        f"Інтеграл: {numerical_integral:.6f} (Похибка: {numerical_error:.2e})"
    )

    # Тестуємо інтеграл Монте-Карло з різними розмірами вибірки
    print("\nОцінки Методом Монте-Карло:")
    monte_carlo_samples = [100, 1000, 10000, 100000, 1000000]

    for sample in monte_carlo_samples:
        mc_result = mc_integral(a, b, num_samples=sample)
        print(f"Інтеграл ({sample:>7,} зразків): {mc_result:.6f}")

    # Будуємо графік функції та інтегрованої області
    x_plot = np.linspace(a - 0.5, b + 0.5, 400)
    y_plot = f(x_plot)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Будуємо графік функції
    ax.plot(x_plot, y_plot, "r", linewidth=2, label="$f(x) = x^2 + 4x - 3$")

    # Заповнюємо область під кривою між a та b
    ix_fill = np.linspace(a, b)
    iy_fill = f(ix_fill)
    ax.fill_between(
        ix_fill, iy_fill, color="gray", alpha=0.4, label="Інтегрована Область"
    )

    # Встановлюємо межі та мітки осей
    ax.set_xlim([x_plot[0], x_plot[-1]])
    ax.set_ylim([-5, max(y_plot) + 5])  # Відкоригована межа y для кращої візуалізації
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")

    # Додаємо вертикальні лінії для меж інтегрування
    ax.axvline(x=a, color="green", linestyle="--", label=f"Нижня Межа (a={a})")
    ax.axvline(x=b, color="blue", linestyle="--", label=f"Верхня Межа (b={b})")

    ax.set_title(f"Графічне Інтегрування $f(x) = x^2 + 4x - 3$ від $x={a}$ до $x={b}$")
    ax.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()
