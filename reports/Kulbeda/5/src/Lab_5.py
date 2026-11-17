import numpy as np
import matplotlib.pyplot as plt

# параметры
a, b, c, d = 0.4, 0.4, 0.08, 0.4
n_in = 6
n_hid = 2
epochs = 2000

# генерация данных
n_train_pts, n_test_pts = 100, 30
x = np.arange(n_train_pts + n_test_pts) * 0.5
y = a * np.cos(b * x) + c * np.sin(d * x)
y_train, y_test = y[:n_train_pts], y[n_train_pts:]

# обучающая выборка по окнам
Xtr = np.array([y_train[i-n_in:i] for i in range(n_in, n_train_pts)])
Ytr = y_train[n_in:]

# сигмоида
def sig(z):
    return 1 / (1 + np.exp(-z))

# обучение
def train(alpha):
    np.random.seed(1)
    W1 = np.random.randn(n_in, n_hid) * 0.5
    b1 = np.zeros(n_hid)
    W2 = np.random.randn(n_hid, 1) * 0.5
    b2 = np.zeros(1)
    errs = []
    for ep in range(epochs):
        # прямой проход
        A1 = sig(Xtr @ W1 + b1)
        pred = (A1 @ W2 + b2).flatten()
        errs.append(np.mean((pred - Ytr) ** 2))
        # обратный проход
        dZ2 = 2 * (pred - Ytr) / len(Ytr)
        dW2 = A1.T @ dZ2.reshape(-1, 1)
        db2 = dZ2.sum()
        dA1 = dZ2.reshape(-1, 1) @ W2.T
        dZ1 = dA1 * A1 * (1 - A1)
        dW1 = Xtr.T @ dZ1
        db1 = dZ1.sum(axis=0)
        W1 -= alpha * dW1
        b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * db2
    return (W1, b1, W2, b2), errs

def predict(params, X):
    W1, b1, W2, b2 = params
    return (sig(X @ W1 + b1) @ W2 + b2).flatten()

# многошаговый прогноз
def forecast(params, history, steps):
    window = list(history[-n_in:])
    out = []
    for _ in range(steps):
        v = predict(params, np.array(window[-n_in:]).reshape(1, -1))[0]
        out.append(v)
        window.append(v)
    return np.array(out)


# оформление таблиц
def print_table(title, headers, rows, fmts):
    cells = [[f.format(v) for f, v in zip(fmts, r)] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    row = lambda vals: "  ".join(f"{v:>{x}}" for v, x in zip(vals, w))
    sep = "-" * (sum(w) + 2 * (len(w) - 1))
    print("\n" + title)
    print(sep)
    print(row(headers))
    print(sep)
    for r in cells:
        print(row(r))
    print(sep)

# подбираем alpha по ошибке обучения
alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
best, best_err = None, np.inf
alpha_rows = []
with np.errstate(all="ignore"):
    for al in alphas:
        params, losses = train(al)
        fc = forecast(params, y_train, n_test_pts)
        alpha_rows.append((al, losses[-1], np.mean((fc - y_test) ** 2)))
        if np.isfinite(losses[-1]) and losses[-1] < best_err:
            best_err, best = losses[-1], al

print_table("ПОДБОР СКОРОСТИ ОБУЧЕНИЯ", ["alpha", "MSE обучения", "MSE прогноза"],
            alpha_rows, ["{:g}", "{:.3e}", "{:.3e}"])
print("\nлучшая alpha =", best)
params, losses = train(best)
ptr = predict(params, Xtr)
pte = forecast(params, y_train, n_test_pts)

# таблицы
def make_rows(ref, got):
    return [(i + 1, r, g, g - r) for i, (r, g) in enumerate(zip(ref, got))]

hdr = ["№", "Эталон", "Получено", "Отклонение"]
fm = ["{:d}", "{:.6f}", "{:.6f}", "{:+.2e}"]
print_table("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ", hdr, make_rows(Ytr, ptr), fm)
print(f"MSE = {np.mean((ptr - Ytr) ** 2):.3e}   MAE = {np.mean(np.abs(ptr - Ytr)):.3e}")
print_table("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ", hdr, make_rows(y_test, pte), fm)
print(f"MSE = {np.mean((pte - y_test) ** 2):.3e}   MAE = {np.mean(np.abs(pte - y_test)):.3e}")

# графики
plt.figure()
plt.plot(x[:n_train_pts], y_train, 'k-', label='функция (обучение)')
plt.plot(x[n_in:n_train_pts], ptr, 'b.', label='ИНС (обучение)')
plt.legend(); plt.grid(); plt.title('участок обучения'); plt.show()

plt.figure()
plt.plot(x[n_train_pts:], y_test, 'k-o', label='эталон')
plt.plot(x[n_train_pts:], pte, 'r.--', label='прогноз')
plt.legend(); plt.grid(); plt.title('прогноз'); plt.show()

plt.figure()
plt.semilogy(losses)
plt.title('ошибка по итерациям'); plt.xlabel('эпоха'); plt.ylabel('MSE')
plt.grid(); plt.show()