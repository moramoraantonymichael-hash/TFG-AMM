
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf


np.random.seed(42)


# Datos y rendimientos

ticker = "^GSPC"
sp500 = yf.download(
    ticker,
    start="2010-01-01",
    end="2023-12-31",
    progress=False,
)

print(sp500.head())
print(f"\nDimensiones: {sp500.shape}")

if isinstance(sp500.columns, pd.MultiIndex):
    precio = sp500["Close"][ticker].dropna()
else:
    precio = sp500["Close"].dropna()

precio = precio.astype(float)
precio.name = "SP500"

print(precio.head())
print(
    f"\nDesde: {precio.index[0].date()} "
    f"Hasta: {precio.index[-1].date()}"
)
print(f"Total sesiones: {len(precio)}")

log_ret = np.log(precio / precio.shift(1)).dropna()
log_ret.name = "log_rendimientos"

print(log_ret.head(10))
print(f"\nTotal rendimientos: {len(log_ret)}")
print(
    f"Peor día: {log_ret.idxmin().date()} → "
    f"{log_ret.min():.4f} ({log_ret.min() * 100:.2f}%)"
)
print(
    f"Mejor día: {log_ret.idxmax().date()} → "
    f"{log_ret.max():.4f} ({log_ret.max() * 100:.2f}%)"
)


# Estadística descriptiva

media_diaria = log_ret.mean()
volatilidad_dia = log_ret.std()
volatilidad_anual = volatilidad_dia * np.sqrt(252)
skewness = log_ret.skew()
kurtosis = log_ret.kurt()
minimo = log_ret.min()
maximo = log_ret.max()

tabla = pd.DataFrame(
    {
        "Estadístico": [
            "Media diaria",
            "Volatilidad diaria",
            "Volatilidad anual",
            "Skewness",
            "Kurtosis (exceso)",
            "Mínimo (peor día)",
            "Máximo (mejor día)",
        ],
        "Valor": [
            f"{media_diaria:.6f}",
            f"{volatilidad_dia:.6f}",
            f"{volatilidad_anual:.4f}",
            f"{skewness:.4f}",
            f"{kurtosis:.4f}",
            f"{minimo:.4f}",
            f"{maximo:.4f}",
        ],
        "Interpretación": [
            "~0.04% diario → crecimiento lento y sostenido",
            "Dispersión típica de un día",
            "Riesgo anual ~15-20% esperado",
            "Negativo → más caídas extremas que subidas",
            "> 0 → colas más gruesas que la normal",
            "Crash de COVID-19 en marzo de 2020",
            "Rebote de COVID-19 en marzo de 2020",
        ],
    }
)

print(tabla.to_string(index=False))


# Distribución de rendimientos

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.6,
    label="Rendimientos reales S&P 500",
)

x = np.linspace(log_ret.min(), log_ret.max(), 300)
curva_normal = stats.norm.pdf(x, media_diaria, volatilidad_dia)

ax.plot(
    x,
    curva_normal,
    color="red",
    linewidth=2,
    label=(
        "Normal teórica (GBM)\n"
        f"μ={media_diaria:.5f}, σ={volatilidad_dia:.5f}"
    ),
)
ax.set_title(
    "Distribución de log-rendimientos diarios del S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("histograma_rendimientos.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.6,
    label="Rendimientos reales S&P 500",
)
ax.plot(
    x,
    curva_normal,
    color="red",
    linewidth=2,
    label=(
        "Normal teórica (GBM)\n"
        f"μ={media_diaria:.5f}, σ={volatilidad_dia:.5f}"
    ),
)
ax.set_title(
    "Distribución de log-rendimientos diarios del S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.set_xlim(-0.03, 0.03)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("histograma_rendimientos_zoom.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.6,
    label="Datos reales",
)
ax.plot(
    x,
    curva_normal,
    color="darkred",
    linewidth=2,
    label="Normal (GBM)",
)
ax.set_xlim(-0.05, 0.00)
ax.set_yscale("log")
ax.set_ylim(1e-3, 1e2)
ax.set_title(
    "Cola izquierda (escala log)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cola_izquierda_log.png", dpi=150)
plt.show()


# QQ-plot

fig, ax = plt.subplots(figsize=(7, 7))

(osm, osr), (slope, intercept, _) = stats.probplot(
    log_ret,
    dist="norm",
)

ax.scatter(
    osm,
    osr,
    color="steelblue",
    alpha=0.4,
    s=8,
    label="Datos reales",
)

x_line = np.array([osm[0], osm[-1]])

ax.plot(
    x_line,
    slope * x_line + intercept,
    color="red",
    linewidth=2,
    label="Normal teórica (GBM)",
)
ax.set_title(
    "QQ-plot: Log-rendimientos S&P 500 vs Normal (GBM)\n"
    "2010–2023",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Cuantiles teóricos (Normal)", fontsize=11)
ax.set_ylabel("Cuantiles empíricos (datos reales)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("qqplot.png", dpi=150)
plt.show()


# Test de Jarque-Bera

jb_stat, jb_pvalue = jarque_bera(log_ret)

print("=" * 45)
print("       TEST DE JARQUE-BERA")
print("=" * 45)
print("  H0: los rendimientos son normales")
print("  H1: los rendimientos no son normales")
print("-" * 45)
print(f"  Estadístico JB : {jb_stat:,.2f}")
print(f"  P-valor        : {jb_pvalue:.2e}")
print("-" * 45)

if jb_pvalue < 0.05:
    print("  CONCLUSIÓN: rechazamos H0 al 95%")
    print("  Los rendimientos no son normales.")
    print("  El GBM queda formalmente refutado.")
else:
    print("  CONCLUSIÓN: no se rechaza H0 al 95%.")

print("=" * 45)


# Autocorrelación

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 9),
)

plot_acf(
    log_ret,
    lags=40,
    ax=axes[0],
    zero=False,
    color="steelblue",
    vlines_kwargs={"colors": "steelblue"},
)

axes[0].set_title(
    "ACF de rendimientos $r_t$ — S&P 500 (2010–2023)\n"
    "Autocorrelación generalmente reducida",
    fontsize=12,
    fontweight="bold",
)
axes[0].set_xlabel("Lag (días)", fontsize=11)
axes[0].set_ylabel("Autocorrelación", fontsize=11)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].grid(True, alpha=0.3)

plot_acf(
    log_ret**2,
    lags=40,
    ax=axes[1],
    zero=False,
    color="darkorange",
    vlines_kwargs={"colors": "darkorange"},
)

axes[1].set_title(
    "ACF de rendimientos al cuadrado $r_t^2$ — S&P 500 "
    "(2010–2023)\n"
    "Evidencia de clustering de volatilidad",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlabel("Lag (días)", fontsize=11)
axes[1].set_ylabel("Autocorrelación", fontsize=11)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    "acf_rendimientos_combinada.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()


# Estimación de parámetros del GBM

mu_diario = log_ret.mean()
sigma_diario = log_ret.std()
mu_anual = mu_diario * 252
sigma_anual = sigma_diario * np.sqrt(252)
n = len(log_ret)
error_estandar = sigma_diario / np.sqrt(n)
z_95 = 1.96
ic_inf = (mu_diario - z_95 * error_estandar) * 252
ic_sup = (mu_diario + z_95 * error_estandar) * 252

print("=" * 50)
print("   Parámetros estimados del GBM")
print("=" * 50)
print(f"  Observaciones        : {n}")
print("-" * 50)
print(f"  μ diario             : {mu_diario:.6f}")
print(
    f"  μ anualizado         : {mu_anual:.4f} "
    f"({mu_anual * 100:.2f}%)"
)
print(f"  IC 95% μ anual       : [{ic_inf:.4f}, {ic_sup:.4f}]")
print("-" * 50)
print(f"  σ diario             : {sigma_diario:.6f}")
print(
    f"  σ anualizado         : {sigma_anual:.4f} "
    f"({sigma_anual * 100:.2f}%)"
)
print("=" * 50)


# Simulación del GBM

np.random.seed(42)

n_sim = 10_000
n_dias = len(log_ret)
dt = 1 / 252
s0 = float(precio.iloc[1])

z = np.random.standard_normal(size=(n_sim, n_dias))
incrementos = np.exp(
    mu_anual * dt
    + sigma_anual * np.sqrt(dt) * z
)
simulaciones = s0 * np.cumprod(incrementos, axis=1)

print(f"Matriz de simulaciones: {simulaciones.shape}")
print(f"Precio inicial S0     : {s0:.2f}")
print(f"Precio final real     : {precio.iloc[-1]:.2f}")
print(
    "Precio final simulado (media): "
    f"{simulaciones[:, -1].mean():.2f}"
)

precio_sim = precio.iloc[1:]

fig, ax = plt.subplots(figsize=(12, 6))

for i in range(20):
    ax.plot(
        precio_sim.index,
        simulaciones[i],
        color="gray",
        alpha=0.4,
        linewidth=0.8,
    )

ax.plot(
    precio_sim.index,
    precio_sim,
    color="steelblue",
    linewidth=2,
    label="Precio real S&P 500",
)

media_tray = simulaciones.mean(axis=0)

ax.plot(
    precio_sim.index,
    media_tray,
    color="red",
    linewidth=1.5,
    linestyle="--",
    label="Media de 10.000 simulaciones GBM",
)
ax.set_title(
    "Simulación Monte Carlo GBM vs Precio Real\n"
    "S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Fecha", fontsize=11)
ax.set_ylabel("Precio ($)", fontsize=11)
ax.set_ylim(0, 10_000)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("trayectorias_gbm.png", dpi=150)
plt.show()

log_ret_sim = np.diff(np.log(simulaciones), axis=1)
todos_sim = log_ret_sim.ravel()
todos_sim_serie = pd.Series(todos_sim)

print("=" * 60)
print("   Estadísticos simulados vs reales")
print("=" * 60)
print(f"{'Estadístico':<25} {'Real':>10} {'GBM simulado':>15}")
print("-" * 60)
print(
    f"{'Media diaria':<25} "
    f"{log_ret.mean():>10.6f} "
    f"{todos_sim.mean():>15.6f}"
)
print(
    f"{'Volatilidad diaria':<25} "
    f"{log_ret.std():>10.6f} "
    f"{todos_sim.std():>15.6f}"
)
print(
    f"{'Skewness':<25} "
    f"{log_ret.skew():>10.4f} "
    f"{todos_sim_serie.skew():>15.4f}"
)
print(
    f"{'Kurtosis (exceso)':<25} "
    f"{log_ret.kurt():>10.4f} "
    f"{todos_sim_serie.kurt():>15.4f}"
)
print(
    f"{'Mínimo':<25} "
    f"{log_ret.min():>10.4f} "
    f"{todos_sim.min():>15.4f}"
)
print(
    f"{'Máximo':<25} "
    f"{log_ret.max():>10.4f} "
    f"{todos_sim.max():>15.4f}"
)
print("=" * 60)

kurtosis_sim = pd.DataFrame(log_ret_sim).kurt(axis=1)

print(f"Kurtosis media simulada : {kurtosis_sim.mean():.4f}")
print(f"Kurtosis mínima simulada: {kurtosis_sim.min():.4f}")
print(f"Kurtosis máxima simulada: {kurtosis_sim.max():.4f}")
print(f"Kurtosis real S&P 500   : {kurtosis:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    kurtosis_sim,
    bins=40,
    color="steelblue",
    alpha=0.7,
    label="Kurtosis de 10.000 simulaciones GBM",
)
ax.axvline(
    kurtosis,
    color="red",
    linewidth=2.5,
    label=f"Kurtosis real S&P 500 = {kurtosis:.2f}",
)
ax.axvline(
    kurtosis_sim.mean(),
    color="orange",
    linewidth=1.5,
    linestyle="--",
    label=f"Media simulada = {kurtosis_sim.mean():.4f}",
)
ax.set_title(
    "Distribución de la kurtosis simulada por el GBM\n"
    "vs kurtosis real del S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Kurtosis (exceso)", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kurtosis_simulada.png", dpi=150)
plt.show()


# Modelo de Heston

def simular_heston(
    mu,
    v0,
    kappa,
    theta,
    xi,
    rho,
    n_sim,
    n_dias,
    dt,
):
    """Simula log-rendimientos con el modelo de Heston."""

    v = np.zeros((n_sim, n_dias))
    s = np.zeros((n_sim, n_dias))

    v[:, 0] = v0
    s[:, 0] = 1.0

    for t in range(1, n_dias):
        z1 = np.random.standard_normal(n_sim)
        z2 = np.random.standard_normal(n_sim)
        zv = rho * z1 + np.sqrt(1 - rho**2) * z2
        v_pos = np.maximum(v[:, t - 1], 0)

        v[:, t] = (
            v_pos
            + kappa * (theta - v_pos) * dt
            + xi * np.sqrt(v_pos * dt) * zv
        )

        s[:, t] = s[:, t - 1] * np.exp(
            (mu - 0.5 * v_pos) * dt
            + np.sqrt(v_pos * dt) * z1
        )

    return np.diff(np.log(s), axis=1)


def calcular_error_calibracion(
    simulacion,
    media_objetivo,
    varianza_objetivo,
    kurtosis_objetivo,
):
    todos = simulacion.ravel()
    media_simulada = np.mean(todos)
    varianza_simulada = np.var(todos)
    kurtosis_simulada = pd.Series(todos).kurt()

    error_media = (
        (media_simulada - media_objetivo) / media_objetivo
    ) ** 2
    error_varianza = (
        (varianza_simulada - varianza_objetivo)
        / varianza_objetivo
    ) ** 2
    error_kurtosis = (
        (kurtosis_simulada - kurtosis_objetivo)
        / kurtosis_objetivo
    ) ** 2

    error_total = error_media + error_varianza + error_kurtosis

    return error_total, kurtosis_simulada


def calibrar_heston(
    v0_grid,
    theta_grid,
    xi_grid,
    mu,
    kappa,
    rho,
    n_dias,
    dt,
    media_objetivo,
    varianza_objetivo,
    kurtosis_objetivo,
    n_sim=200,
    aplicar_feller=False,
):
    mejor_error = np.inf
    mejores_parametros = None
    resultados = []
    combinaciones_validas = 0

    for v0 in v0_grid:
        for theta in theta_grid:
            for xi in xi_grid:
                valor_feller = 2 * kappa * theta - xi**2

                if aplicar_feller and valor_feller <= 0:
                    continue

                combinaciones_validas += 1

                simulacion = simular_heston(
                    mu=mu,
                    v0=v0,
                    kappa=kappa,
                    theta=theta,
                    xi=xi,
                    rho=rho,
                    n_sim=n_sim,
                    n_dias=n_dias,
                    dt=dt,
                )

                error, kurtosis_simulada = calcular_error_calibracion(
                    simulacion=simulacion,
                    media_objetivo=media_objetivo,
                    varianza_objetivo=varianza_objetivo,
                    kurtosis_objetivo=kurtosis_objetivo,
                )

                resultado = {
                    "v0": v0,
                    "theta": theta,
                    "xi": xi,
                    "kurtosis_sim": round(kurtosis_simulada, 4),
                    "error": round(error, 6),
                }

                if aplicar_feller:
                    resultado["feller"] = round(valor_feller, 4)

                resultados.append(resultado)

                if error < mejor_error:
                    mejor_error = error
                    mejores_parametros = (v0, theta, xi)

    return (
        mejores_parametros,
        mejor_error,
        pd.DataFrame(resultados),
        combinaciones_validas,
    )


# Calibración inicial de Heston

np.random.seed(42)

rho = -0.7
kappa = 2.0
media_obj = log_ret.mean()
varianza_obj = log_ret.var()
kurtosis_obj = log_ret.kurt()

v0_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
theta_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
xi_grid = [0.2, 0.4, 0.6, 0.8, 1.0]

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(f"Probando {total} combinaciones...")

(
    mejor_params,
    mejor_error,
    resultados,
    _,
) = calibrar_heston(
    v0_grid=v0_grid,
    theta_grid=theta_grid,
    xi_grid=xi_grid,
    mu=mu_anual,
    kappa=kappa,
    rho=rho,
    n_dias=n_dias,
    dt=dt,
    media_objetivo=media_obj,
    varianza_objetivo=varianza_obj,
    kurtosis_objetivo=kurtosis_obj,
)

print("\nMejores parámetros del primer nivel:")
print(
    f"  v0={mejor_params[0]}, "
    f"theta={mejor_params[1]}, "
    f"xi={mejor_params[2]}"
)
print(f"  Error total = {mejor_error:.6f}")
print(
    resultados.sort_values("error")
    .head(5)
    .to_string(index=False)
)

np.random.seed(42)

v0_grid = [0.03, 0.035, 0.04, 0.045, 0.05]
theta_grid = [0.015, 0.02, 0.025, 0.03]
xi_grid = [0.7, 0.75, 0.8, 0.85, 0.9]

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(f"\nProbando {total} combinaciones en el grid refinado...")

(
    mejor_params,
    mejor_error,
    resultados_refinados,
    _,
) = calibrar_heston(
    v0_grid=v0_grid,
    theta_grid=theta_grid,
    xi_grid=xi_grid,
    mu=mu_anual,
    kappa=kappa,
    rho=rho,
    n_dias=n_dias,
    dt=dt,
    media_objetivo=media_obj,
    varianza_objetivo=varianza_obj,
    kurtosis_objetivo=kurtosis_obj,
)

print("\nMejores parámetros refinados:")
print(
    f"  v0={mejor_params[0]}, "
    f"theta={mejor_params[1]}, "
    f"xi={mejor_params[2]}"
)
print(f"  rho={rho} (fijo), kappa={kappa} (fijo)")
print(f"  Error total = {mejor_error:.6f}")
print(
    resultados_refinados.sort_values("error")
    .head(5)
    .to_string(index=False)
)


# Simulación final de Heston

np.random.seed(42)

v0_final, theta_final, xi_final = mejor_params

print(mejor_params)
print("Simulando Heston con parámetros calibrados...")

log_ret_heston = simular_heston(
    mu=mu_anual,
    v0=v0_final,
    kappa=kappa,
    theta=theta_final,
    xi=xi_final,
    rho=rho,
    n_sim=1_000,
    n_dias=n_dias,
    dt=dt,
)

todos_heston = log_ret_heston.ravel()
todos_heston_serie = pd.Series(todos_heston)

print("\n" + "=" * 65)
print("   COMPARACIÓN FINAL: Real vs GBM vs Heston")
print("=" * 65)
print(
    f"{'Estadístico':<25} "
    f"{'Real':>10} "
    f"{'GBM':>12} "
    f"{'Heston':>12}"
)
print("-" * 65)
print(
    f"{'Media diaria':<25} "
    f"{log_ret.mean():>10.6f} "
    f"{todos_sim.mean():>12.6f} "
    f"{todos_heston.mean():>12.6f}"
)
print(
    f"{'Volatilidad diaria':<25} "
    f"{log_ret.std():>10.6f} "
    f"{todos_sim.std():>12.6f} "
    f"{todos_heston.std():>12.6f}"
)
print(
    f"{'Skewness':<25} "
    f"{log_ret.skew():>10.4f} "
    f"{todos_sim_serie.skew():>12.4f} "
    f"{todos_heston_serie.skew():>12.4f}"
)
print(
    f"{'Kurtosis (exceso)':<25} "
    f"{log_ret.kurt():>10.4f} "
    f"{todos_sim_serie.kurt():>12.4f} "
    f"{todos_heston_serie.kurt():>12.4f}"
)
print(
    f"{'Mínimo':<25} "
    f"{log_ret.min():>10.4f} "
    f"{todos_sim.min():>12.4f} "
    f"{todos_heston.min():>12.4f}"
)
print(
    f"{'Máximo':<25} "
    f"{log_ret.max():>10.4f} "
    f"{todos_sim.max():>12.4f} "
    f"{todos_heston.max():>12.4f}"
)
print("=" * 65)

fig, ax = plt.subplots(figsize=(12, 6))

x_min, x_max = -0.08, 0.08
x = np.linspace(x_min, x_max, 300)

ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.5,
    label="Rendimientos reales S&P 500",
)
ax.hist(
    todos_heston,
    bins=100,
    density=True,
    color="green",
    alpha=0.4,
    label="Heston simulado",
)

curva_normal = stats.norm.pdf(
    x,
    log_ret.mean(),
    log_ret.std(),
)

ax.plot(
    x,
    curva_normal,
    color="red",
    linewidth=2.5,
    label="Normal teórica (GBM), Kurtosis ≈ 0",
)

ax.text(
    0.02,
    55,
    f"Kurtosis real:   {log_ret.kurt():.2f}",
    color="steelblue",
    fontsize=9,
    fontweight="bold",
)
ax.text(
    0.02,
    50,
    f"Kurtosis Heston: {todos_heston_serie.kurt():.2f}",
    color="green",
    fontsize=9,
    fontweight="bold",
)
ax.text(
    0.02,
    45,
    f"Kurtosis GBM:    {todos_sim_serie.kurt():.2f}",
    color="red",
    fontsize=9,
    fontweight="bold",
)

ax.set_title(
    "Comparación de distribuciones: Real vs GBM vs Heston\n"
    "S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.set_xlim(x_min, x_max)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("comparacion_final.png", dpi=150)
plt.show()


# Calibración de Heston con condición de Feller

np.random.seed(42)

rho = -0.7
kappa = 2.0

v0_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
theta_grid = [0.03, 0.05, 0.07, 0.10, 0.15]
xi_grid = [0.10, 0.15, 0.20, 0.25, 0.30]

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(
    f"Probando {total} combinaciones "
    "en la calibración con Feller..."
)

(
    mejor_params_feller,
    mejor_error_feller,
    resultados_feller,
    combinaciones_validas,
) = calibrar_heston(
    v0_grid=v0_grid,
    theta_grid=theta_grid,
    xi_grid=xi_grid,
    mu=mu_anual,
    kappa=kappa,
    rho=rho,
    n_dias=n_dias,
    dt=dt,
    media_objetivo=media_obj,
    varianza_objetivo=varianza_obj,
    kurtosis_objetivo=kurtosis_obj,
    aplicar_feller=True,
)

v0_best, theta_best, xi_best = mejor_params_feller
feller_best = 2 * kappa * theta_best - xi_best**2

print(
    "Combinaciones que cumplen Feller: "
    f"{combinaciones_validas}/{total}"
)
print("\nMejores parámetros del primer nivel:")
print(
    f"  v0={v0_best}, "
    f"theta={theta_best}, "
    f"xi={xi_best}"
)
print(f"  Feller: 2κθ - ξ² = {feller_best:.4f} > 0")
print(f"  Error total = {mejor_error_feller:.6f}")
print(
    resultados_feller.sort_values("error")
    .head(5)
    .to_string(index=False)
)

v0_grid = [
    max(0.005, v0_best + desplazamiento)
    for desplazamiento in [-0.015, -0.01, 0.0, 0.01, 0.015]
]
theta_grid = [
    max(0.01, theta_best + desplazamiento)
    for desplazamiento in [-0.02, -0.01, 0.0, 0.01, 0.02]
]
xi_grid = [
    max(0.05, xi_best + desplazamiento)
    for desplazamiento in [-0.05, -0.02, 0.0, 0.02, 0.05]
]

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(
    f"\nProbando {total} combinaciones "
    "en el grid refinado con Feller..."
)

(
    mejor_params_feller,
    mejor_error_feller,
    resultados_feller_refinados,
    combinaciones_validas,
) = calibrar_heston(
    v0_grid=v0_grid,
    theta_grid=theta_grid,
    xi_grid=xi_grid,
    mu=mu_anual,
    kappa=kappa,
    rho=rho,
    n_dias=n_dias,
    dt=dt,
    media_objetivo=media_obj,
    varianza_objetivo=varianza_obj,
    kurtosis_objetivo=kurtosis_obj,
    aplicar_feller=True,
)

v0_best, theta_best, xi_best = mejor_params_feller
feller_best = 2 * kappa * theta_best - xi_best**2

print(
    "Combinaciones que cumplen Feller: "
    f"{combinaciones_validas}/{total}"
)
print("\nMejores parámetros refinados:")
print(
    f"  v0={v0_best:.4f}, "
    f"theta={theta_best:.4f}, "
    f"xi={xi_best:.4f}"
)
print(f"  rho={rho} (fijo), kappa={kappa} (fijo)")
print(f"  Feller: 2κθ - ξ² = {feller_best:.4f} > 0")
print(f"  Error total = {mejor_error_feller:.6f}")
print(
    resultados_feller_refinados.sort_values("error")
    .head(5)
    .to_string(index=False)
)

