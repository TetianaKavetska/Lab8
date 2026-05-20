import numpy as np
import matplotlib.pyplot as plt
import cmath

EPS = 1e-10

#ФУНКЦІЇ ТА ТАБУЛЯЦІЯ

def F(x):
    return x ** 3 - 3 * x + 1

def dF(x):
    return 3 * x ** 2 - 3

def ddF(x):
    return 6 * x

def tabulate_and_plot(a, b, h):
    x_vals = np.arange(a, b, h)
    y_vals = F(x_vals)

    #Запис у файл
    with open("table.txt", "w", encoding="utf-8") as f:
        f.write("x\tF(x)\n")
        for x, y in zip(x_vals, y_vals):
            f.write(f"{x:.3f}\t{y:.6f}\n")

    #Побудова графіка функції
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, label="F(x) = x³ - 3x + 1", color='blue', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Вісь X
    plt.axvline(0, color='black', linewidth=1)  # Вісь Y
    plt.title("Графік функції F(x)")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.grid(True)
    plt.legend()
    plt.draw()

    return x_vals, y_vals

def find_intervals(x, y):
    intervals = []
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] < 0:
            intervals.append((x[i], x[i + 1]))
    return intervals

#МЕТОДИ РОЗВ'ЯЗКУ

def simple_iteration(x0):
    tau, n, log = -0.1, 0, [x0]
    for _ in range(1000):
        x1 = x0 + tau * F(x0)
        log.append(x1);
        n += 1
        if abs(x1 - x0) < EPS: break
        x0 = x1
    return x0, n, log


def newton(x0):
    n, log = 0, [x0]
    for _ in range(1000):
        df = dF(x0)
        if abs(df) < 1e-12: break
        x1 = x0 - F(x0) / df
        log.append(x1);
        n += 1
        if abs(x1 - x0) < EPS: break
        x0 = x1
    return x0, n, log


def chebyshev(x0):
    n, log = 0, [x0]
    for _ in range(1000):
        df, fv = dF(x0), F(x0)
        if abs(df) < 1e-12: break
        x1 = x0 - fv / df - (ddF(x0) * fv ** 2) / (2 * df ** 3)
        log.append(x1);
        n += 1
        if abs(x1 - x0) < EPS: break
        x0 = x1
    return x0, n, log


def secant(x0, x1):
    n, log = 0, [x0, x1]
    for _ in range(1000):
        f1, f0 = F(x1), F(x0)
        if abs(f1 - f0) < 1e-12: break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        log.append(x2);
        n += 1
        if abs(x2 - x1) < EPS: break
        x0, x1 = x1, x2
    return x1, n, log


def parabola_method(x0, x1, x2):
    n, log = 0, [x0, x1, x2]
    for _ in range(1000):
        f0, f1, f2 = F(x0), F(x1), F(x2)
        f01, f12 = (f1 - f0) / (x1 - x0), (f2 - f1) / (x2 - x1)
        f012 = (f12 - f01) / (x2 - x0)
        w = f12 + f012 * (x2 - x1)
        discr = cmath.sqrt(w ** 2 - 4 * f2 * f012)
        denom = w + discr if abs(w + discr) > abs(w - discr) else w - discr
        x3 = x2 - 2 * f2 / denom
        log.append(x3.real);
        n += 1
        if abs(x3.real - x2) < EPS: return x3.real, n, log
        x0, x1, x2 = x1, x2, x3.real
    return x2, n, log


def inverse_interpolation(x0, x1, x2):
    n, log = 0, [x0, x1, x2]
    for _ in range(1000):
        y0, y1, y2 = F(x0), F(x1), F(x2)
        x3 = (x0 * y1 * y2) / ((y0 - y1) * (y0 - y2)) + (x1 * y0 * y2) / ((y1 - y0) * (y1 - y2)) + (x2 * y0 * y1) / (
                    (y2 - y0) * (y2 - y1))
        log.append(x3);
        n += 1
        if abs(x3 - x2) < EPS: return x3, n, log
        x0, x1, x2 = x1, x2, x3
    return x2, n, log

#АЛГЕБРАЇЧНІ
def horner(a, x):
    b = [a[0]]
    for i in range(1, len(a)): b.append(a[i] + b[i - 1] * x)
    return b

def newton_horner(a, x0):
    n = 0
    for _ in range(500):
        b = horner(a, x0)
        c = horner(b[:-1], x0)
        x1 = x0 - b[-1] / c[-1]
        n += 1
        if abs(x1 - x0) < EPS: return x1, n
        x0 = x1
    return x0, n

def lin_method(a, p0, q0):
    n_deg, p, q, n_iter = len(a) - 1, p0, q0, 0
    for _ in range(1000):
        b = [0] * (n_deg + 1)
        b[0] = a[0]
        b[1] = a[1] - p * b[0]
        for i in range(2, n_deg - 1): b[i] = a[i] - p * b[i - 1] - q * b[i - 2]
        R, S = a[n_deg - 1] - q * b[n_deg - 3], a[n_deg]
        p_new, q_new = R / b[n_deg - 2], S / b[n_deg - 2]
        n_iter += 1
        if abs(p_new - p) < EPS and abs(q_new - q) < EPS: break
        p, q = p_new, q_new
    D = p ** 2 - 4 * q
    return (-p + cmath.sqrt(D)) / 2, (-p - cmath.sqrt(D)) / 2, n_iter

#ВИКОНАННЯ
if __name__ == "__main__":

    x_range, y_range = tabulate_and_plot(-3, 3, 0.1)
    intervals = find_intervals(x_range, y_range)
    print(f"Інтервали: {intervals}")

    if intervals:
        a, b = intervals[0]
        results = {
            "Ітерація": simple_iteration(a),
            "Ньютон": newton(a),
            "Чебишев": chebyshev(a),
            "Хорди": secant(a, b),
            "Парабол": parabola_method(a, (a + b) / 2, b),
            "Зворотна": inverse_interpolation(a, (a + b) / 2, b)
        }

        #Графік збіжності методів
        plt.figure(figsize=(10, 6))
        for name, (root, iters, log) in results.items():
            plt.plot(log, label=f"{name} ({iters} іт.)", marker='o', markersize=4)
            print(f"{name:<15} | Корінь: {root:<12.10f} | Ітерацій: {iters}")

        plt.title("Порівняння швидкості збіжності всіх методів")
        plt.xlabel("Номер ітерації")
        plt.ylabel("Значення x")
        plt.legend()
        plt.grid(True)

    poly_coeffs = [1.0, 0.0, -3.0, 1.0]
    rh, ih = newton_horner(poly_coeffs, -2.0)
    z1, z2, il = lin_method(poly_coeffs, 1.0, 1.0)

    print(f"\nНьютон-Горнер: {rh:.10f} (іт: {ih})")
    print(f"Метод Ліна: z1={z1}, z2={z2} (іт: {il})")

    plt.show()
