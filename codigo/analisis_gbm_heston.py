
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf


# Semilla para garantizar la reproducibilidad
np.random.seed(42)


# =============================================================
# 1. DESCARGA Y PREPARACIÓN DE LOS DATOS
# =============================================================

ticker = "^GSPC"

sp500 = yf.download(
    ticker,
    start="2010-01-01",
    end="2023-12-31"
)

print(sp500.head())
print(f"\nDimensiones: {sp500.shape}")


# Manejo del MultiIndex utilizado por algunas versiones de yfinance
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


# Cálculo de los log-rendimientos diarios
log_ret = np.log(precio / precio.shift(1)).dropna()
log_ret.name = "log_rendimientos"

print(log_ret.head(10))
print(f"\nTotal rendimientos: {len(log_ret)}")

print(
    f"Peor día: {log_ret.idxmin().date()} "
    f"→ {log_ret.min():.4f} "
    f"({log_ret.min() * 100:.2f}%)"
)

print(
    f"Mejor día: {log_ret.idxmax().date()} "
    f"→ {log_ret.max():.4f} "
    f"({log_ret.max() * 100:.2f}%)"
)


# =============================================================
# 2. ANÁLISIS ESTADÍSTICO DESCRIPTIVO
# =============================================================

media_diaria = log_ret.mean()
volatilidad_dia = log_ret.std()
volatilidad_anual = volatilidad_dia * np.sqrt(252)
skewness = log_ret.skew()
kurtosis = log_ret.kurt()
minimo = log_ret.min()
maximo = log_ret.max()

tabla = pd.DataFrame({
    "Estadístico": [
        "Media diaria",
        "Volatilidad diaria",
        "Volatilidad anual",
        "Skewness",
        "Kurtosis (exceso)",
        "Mínimo (peor día)",
        "Máximo (mejor día)"
    ],
    "Valor": [
        f"{media_diaria:.6f}",
        f"{volatilidad_dia:.6f}",
        f"{volatilidad_anual:.4f}",
        f"{skewness:.4f}",
        f"{kurtosis:.4f}",
        f"{minimo:.4f}",
        f"{maximo:.4f}"
    ],
    "Interpretación": [
        "Crecimiento medio diario del índice",
        "Dispersión típica de los rendimientos diarios",
        "Riesgo anualizado suponiendo 252 sesiones",
        "Valor negativo: mayor peso de las caídas extremas",
        "Colas más gruesas que las de una distribución normal",
        "Mayor caída diaria del periodo",
        "Mayor subida diaria del periodo"
    ]
})

print(tabla.to_string(index=False))


# Histograma con distribución normal superpuesta
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.6,
    label="Rendimientos reales S&P 500"
)

x = np.linspace(
    log_ret.min(),
    log_ret.max(),
    300
)

curva_normal = stats.norm.pdf(
    x,
    media_diaria,
    volatilidad_dia
)

ax.plot(
    x,
    curva_normal,
    color="red",
    linewidth=2,
    label=(
        "Normal teórica (GBM)\n"
        f"μ={media_diaria:.5f}, "
        f"σ={volatilidad_dia:.5f}"
    )
)

ax.set_title(
    "Distribución de log-rendimientos diarios "
    "del S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Log-rendimiento diario",
    fontsize=11
)

ax.set_ylabel(
    "Densidad",
    fontsize=11
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "histograma_rendimientos.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# Histograma ampliado alrededor del centro de la distribución
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.6,
    label="Rendimientos reales S&P 500"
)

x = np.linspace(
    log_ret.min(),
    log_ret.max(),
    300
)

curva_normal = stats.norm.pdf(
    x,
    media_diaria,
    volatilidad_dia
)

ax.plot(
    x,
    curva_normal,
    color="red",
    linewidth=2,
    label=(
        "Normal teórica (GBM)\n"
        f"μ={media_diaria:.5f}, "
        f"σ={volatilidad_dia:.5f}"
    )
)

ax.set_title(
    "Distribución de log-rendimientos diarios "
    "del S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Log-rendimiento diario",
    fontsize=11
)

ax.set_ylabel(
    "Densidad",
    fontsize=11
)

ax.set_xlim(-0.03, 0.03)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "histograma_rendimientos_zoom.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# Análisis de la cola izquierda con escala logarítmica
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.6,
    label="Datos reales"
)

x = np.linspace(
    log_ret.min(),
    log_ret.max(),
    300
)

curva_normal = stats.norm.pdf(
    x,
    media_diaria,
    volatilidad_dia
)

ax.plot(
    x,
    curva_normal,
    color="darkred",
    linewidth=2,
    label="Normal teórica (GBM)"
)

ax.set_xlim(-0.05, 0.00)
ax.set_yscale("log")
ax.set_ylim(1e-3, 1e2)

ax.set_title(
    "Cola izquierda de la distribución "
    "(escala logarítmica)",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Log-rendimiento diario",
    fontsize=11
)

ax.set_ylabel(
    "Densidad",
    fontsize=11
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "cola_izquierda_log.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# QQ-plot frente a una distribución normal
fig, ax = plt.subplots(figsize=(7, 7))

(osm, osr), (slope, intercept, r) = stats.probplot(
    log_ret,
    dist="norm"
)

ax.scatter(
    osm,
    osr,
    color="steelblue",
    alpha=0.4,
    s=8,
    label="Datos reales"
)

x_line = np.array([
    osm[0],
    osm[-1]
])

ax.plot(
    x_line,
    slope * x_line + intercept,
    color="red",
    linewidth=2,
    label="Normal teórica (GBM)"
)

ax.set_title(
    "QQ-plot: log-rendimientos del S&P 500 "
    "frente a la distribución normal\n"
    "Periodo 2010–2023",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Cuantiles teóricos de la distribución normal",
    fontsize=11
)

ax.set_ylabel(
    "Cuantiles empíricos de los datos reales",
    fontsize=11
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "qqplot.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# Test de normalidad de Jarque-Bera
jb_resultado = jarque_bera(log_ret)
jb_stat = jb_resultado.statistic
jb_pvalue = jb_resultado.pvalue

print("=" * 50)
print("TEST DE JARQUE-BERA")
print("=" * 50)
print("H0: los rendimientos siguen una distribución normal")
print("H1: los rendimientos no siguen una distribución normal")
print("-" * 50)
print(f"Estadístico JB: {jb_stat:,.2f}")
print(f"P-valor: {jb_pvalue:.2e}")
print("-" * 50)

if jb_pvalue < 0.05:
    print("Conclusión: se rechaza H0 al nivel del 5%.")
    print("Los rendimientos no siguen una distribución normal.")
else:
    print("Conclusión: no se rechaza H0 al nivel del 5%.")

print("=" * 50)


# Funciones de autocorrelación
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 9)
)

plot_acf(
    log_ret,
    lags=40,
    ax=axes[0],
    zero=False,
    color="steelblue",
    vlines_kwargs={
        "colors": "steelblue"
    }
)

axes[0].set_title(
    "ACF de los rendimientos $r_t$ — "
    "S&P 500 (2010–2023)\n"
    "Autocorrelación generalmente reducida",
    fontsize=12,
    fontweight="bold"
)

axes[0].set_xlabel(
    "Retardo temporal (días)",
    fontsize=11
)

axes[0].set_ylabel(
    "Autocorrelación",
    fontsize=11
)

axes[0].axhline(
    0,
    color="black",
    linewidth=0.8
)

axes[0].grid(
    True,
    alpha=0.3
)


plot_acf(
    log_ret ** 2,
    lags=40,
    ax=axes[1],
    zero=False,
    color="darkorange",
    vlines_kwargs={
        "colors": "darkorange"
    }
)

axes[1].set_title(
    "ACF de los rendimientos al cuadrado "
    "$r_t^2$ — S&P 500 (2010–2023)\n"
    "Evidencia de agrupamiento de volatilidad",
    fontsize=12,
    fontweight="bold"
)

axes[1].set_xlabel(
    "Retardo temporal (días)",
    fontsize=11
)

axes[1].set_ylabel(
    "Autocorrelación",
    fontsize=11
)

axes[1].axhline(
    0,
    color="black",
    linewidth=0.8
)

axes[1].grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "acf_rendimientos_combinada.png",
    dpi=300,
    bbox_inches="tight",
    format="png"
)

plt.show()


# =============================================================
# 3. ESTIMACIÓN DE LOS PARÁMETROS DEL GBM
# =============================================================

mu_diario = log_ret.mean()
sigma_diario = log_ret.std()

mu_anual = mu_diario * 252
sigma_anual = sigma_diario * np.sqrt(252)

n = len(log_ret)

error_estandar = sigma_diario / np.sqrt(n)
z_95 = 1.96

ic_inf = (
    mu_diario
    - z_95 * error_estandar
) * 252

ic_sup = (
    mu_diario
    + z_95 * error_estandar
) * 252

print("=" * 55)
print("PARÁMETROS ESTIMADOS DEL MODELO GBM")
print("=" * 55)
print(f"Observaciones: {n}")
print("-" * 55)
print(f"μ diario: {mu_diario:.6f}")
print(
    f"μ anualizado: {mu_anual:.4f} "
    f"({mu_anual * 100:.2f}%)"
)
print(
    f"IC del 95% para μ anual: "
    f"[{ic_inf:.4f}, {ic_sup:.4f}]"
)
print("-" * 55)
print(f"σ diario: {sigma_diario:.6f}")
print(
    f"σ anualizado: {sigma_anual:.4f} "
    f"({sigma_anual * 100:.2f}%)"
)
print("=" * 55)


# =============================================================
# 4. SIMULACIÓN DEL MODELO GBM
# =============================================================

N_sim = 100000
N_dias = len(log_ret)
dt = 1 / 252

S0 = float(precio.iloc[1])

Z = np.random.standard_normal(
    size=(N_sim, N_dias)
)

incrementos = np.exp(
    mu_anual * dt
    + sigma_anual * np.sqrt(dt) * Z
)

simulaciones = S0 * np.cumprod(
    incrementos,
    axis=1
)

print(
    f"Matriz de simulaciones: "
    f"{simulaciones.shape}"
)

print(
    f"Precio inicial S0: "
    f"{S0:.2f}"
)

print(
    f"Precio final real: "
    f"{precio.iloc[-1]:.2f}"
)

print(
    "Precio final simulado medio: "
    f"{simulaciones[:, -1].mean():.2f}"
)


# Trayectorias simuladas frente al precio real
precio_sim = precio.iloc[1:]

fig, ax = plt.subplots(figsize=(12, 6))

for i in range(20):
    ax.plot(
        precio_sim.index,
        simulaciones[i],
        color="gray",
        alpha=0.4,
        linewidth=0.8
    )

ax.plot(
    precio_sim.index,
    precio_sim,
    color="steelblue",
    linewidth=2,
    label="Precio real del S&P 500"
)

media_trayectorias = simulaciones.mean(
    axis=0
)

ax.plot(
    precio_sim.index,
    media_trayectorias,
    color="red",
    linewidth=1.5,
    linestyle="--",
    label="Media de 100.000 simulaciones GBM"
)

ax.set_title(
    "Simulación Monte Carlo GBM frente al precio real\n"
    "S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Fecha",
    fontsize=11
)

ax.set_ylabel(
    "Precio",
    fontsize=11
)

ax.set_ylim(0, 10000)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "trayectorias_gbm.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# Estadísticos de las simulaciones GBM
log_ret_sim = np.diff(
    np.log(simulaciones),
    axis=1
)

todos_sim = log_ret_sim.flatten()

print("=" * 65)
print("ESTADÍSTICOS REALES FRENTE AL MODELO GBM")
print("=" * 65)

print(
    f"{'Estadístico':<25}"
    f"{'Real':>12}"
    f"{'GBM simulado':>18}"
)

print("-" * 65)

print(
    f"{'Media diaria':<25}"
    f"{log_ret.mean():>12.6f}"
    f"{todos_sim.mean():>18.6f}"
)

print(
    f"{'Volatilidad diaria':<25}"
    f"{log_ret.std():>12.6f}"
    f"{todos_sim.std():>18.6f}"
)

print(
    f"{'Skewness':<25}"
    f"{log_ret.skew():>12.4f}"
    f"{pd.Series(todos_sim).skew():>18.4f}"
)

print(
    f"{'Kurtosis (exceso)':<25}"
    f"{log_ret.kurt():>12.4f}"
    f"{pd.Series(todos_sim).kurt():>18.4f}"
)

print(
    f"{'Mínimo':<25}"
    f"{log_ret.min():>12.4f}"
    f"{todos_sim.min():>18.4f}"
)

print(
    f"{'Máximo':<25}"
    f"{log_ret.max():>12.4f}"
    f"{todos_sim.max():>18.4f}"
)

print("=" * 65)


# Kurtosis calculada para cada trayectoria simulada
kurtosis_sim = pd.DataFrame(
    log_ret_sim
).kurt(
    axis=1
)

print(
    f"Kurtosis media simulada: "
    f"{kurtosis_sim.mean():.4f}"
)

print(
    f"Kurtosis mínima simulada: "
    f"{kurtosis_sim.min():.4f}"
)

print(
    f"Kurtosis máxima simulada: "
    f"{kurtosis_sim.max():.4f}"
)

print(
    f"Kurtosis real del S&P 500: "
    f"{kurtosis:.4f}"
)


# Histograma de la kurtosis simulada
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    kurtosis_sim,
    bins=40,
    color="steelblue",
    alpha=0.7,
    label="Kurtosis de 100.000 simulaciones GBM"
)

ax.axvline(
    kurtosis,
    color="red",
    linewidth=2.5,
    label=(
        "Kurtosis real del S&P 500 "
        f"= {kurtosis:.2f}"
    )
)

ax.axvline(
    kurtosis_sim.mean(),
    color="orange",
    linewidth=1.5,
    linestyle="--",
    label=(
        "Media simulada "
        f"= {kurtosis_sim.mean():.4f}"
    )
)

ax.set_title(
    "Distribución de la kurtosis simulada por el GBM\n"
    "frente a la kurtosis real del S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Kurtosis en exceso",
    fontsize=11
)

ax.set_ylabel(
    "Frecuencia",
    fontsize=11
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "kurtosis_simulada.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# =============================================================
# 5. MODELO DE HESTON
# =============================================================

def simular_heston(
    mu,
    v0,
    kappa,
    theta,
    xi,
    rho,
    N_sim,
    N_dias,
    dt
):
    """
    Simula trayectorias del modelo de Heston mediante
    el método de Euler-Maruyama.

    Devuelve una matriz de log-rendimientos simulados.
    """

    v = np.zeros(
        (N_sim, N_dias)
    )

    S = np.zeros(
        (N_sim, N_dias)
    )

    v[:, 0] = v0
    S[:, 0] = 1.0

    for t in range(1, N_dias):
        Z1 = np.random.standard_normal(
            N_sim
        )

        Z2 = np.random.standard_normal(
            N_sim
        )

        Zv = (
            rho * Z1
            + np.sqrt(1 - rho ** 2) * Z2
        )

        v_pos = np.maximum(
            v[:, t - 1],
            0
        )

        v[:, t] = (
            v_pos
            + kappa
            * (theta - v_pos)
            * dt
            + xi
            * np.sqrt(v_pos)
            * np.sqrt(dt)
            * Zv
        )

        S[:, t] = (
            S[:, t - 1]
            * np.exp(
                (mu - 0.5 * v_pos) * dt
                + np.sqrt(v_pos)
                * np.sqrt(dt)
                * Z1
            )
        )

    return np.diff(
        np.log(S),
        axis=1
    )


# =============================================================
# 6. CALIBRACIÓN DEL MODELO DE HESTON
# =============================================================

rho = -0.7
kappa = 2.0

media_obj = log_ret.mean()
varianza_obj = log_ret.var()
kurtosis_obj = log_ret.kurt()

v0_grid = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05
]

theta_grid = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05
]

xi_grid = [
    0.2,
    0.4,
    0.6,
    0.8,
    1.0
]

mejor_error = np.inf
mejor_params = None
resultados = []

total = (
    len(v0_grid)
    * len(theta_grid)
    * len(xi_grid)
)

print(
    f"Probando {total} combinaciones "
    "en la primera calibración..."
)

for v0 in v0_grid:
    for theta in theta_grid:
        for xi in xi_grid:

            sim = simular_heston(
                mu=mu_anual,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                N_sim=200,
                N_dias=N_dias,
                dt=dt
            )

            todos = sim.flatten()

            media_sim = np.mean(todos)
            varianza_sim = np.var(todos)

            kurtosis_sim_val = pd.Series(
                todos
            ).kurt()

            error_media = (
                (media_sim - media_obj)
                / media_obj
            ) ** 2

            error_varianza = (
                (varianza_sim - varianza_obj)
                / varianza_obj
            ) ** 2

            error_kurtosis = (
                (kurtosis_sim_val - kurtosis_obj)
                / kurtosis_obj
            ) ** 2

            error_total = (
                error_media
                + error_varianza
                + error_kurtosis
            )

            resultados.append({
                "v0": v0,
                "theta": theta,
                "xi": xi,
                "kurtosis_sim": round(
                    kurtosis_sim_val,
                    4
                ),
                "error": round(
                    error_total,
                    6
                )
            })

            if error_total < mejor_error:
                mejor_error = error_total
                mejor_params = (
                    v0,
                    theta,
                    xi
                )

print("\nMejores parámetros de la primera calibración:")

print(
    f"v0 = {mejor_params[0]}, "
    f"theta = {mejor_params[1]}, "
    f"xi = {mejor_params[2]}"
)

print(
    f"Error total = "
    f"{mejor_error:.6f}"
)

print(
    pd.DataFrame(resultados)
    .sort_values("error")
    .head(5)
    .to_string(index=False)
)


# Grid refinado alrededor de los mejores parámetros
v0_grid = [
    0.03,
    0.035,
    0.04,
    0.045,
    0.05
]

theta_grid = [
    0.015,
    0.02,
    0.025,
    0.03
]

xi_grid = [
    0.7,
    0.75,
    0.8,
    0.85,
    0.9
]

mejor_error = np.inf
mejor_params = None
resultados2 = []

total = (
    len(v0_grid)
    * len(theta_grid)
    * len(xi_grid)
)

print(
    f"\nProbando {total} combinaciones "
    "en el grid refinado..."
)

for v0 in v0_grid:
    for theta in theta_grid:
        for xi in xi_grid:

            sim = simular_heston(
                mu=mu_anual,
                v0=v0,
                kappa=kappa,
                theta=theta,
                xi=xi,
                rho=rho,
                N_sim=200,
                N_dias=N_dias,
                dt=dt
            )

            todos = sim.flatten()

            media_sim = np.mean(todos)
            varianza_sim = np.var(todos)

            kurtosis_sim_val = pd.Series(
                todos
            ).kurt()

            error_media = (
                (media_sim - media_obj)
                / media_obj
            ) ** 2

            error_varianza = (
                (varianza_sim - varianza_obj)
                / varianza_obj
            ) ** 2

            error_kurtosis = (
                (kurtosis_sim_val - kurtosis_obj)
                / kurtosis_obj
            ) ** 2

            error_total = (
                error_media
                + error_varianza
                + error_kurtosis
            )

            resultados2.append({
                "v0": v0,
                "theta": theta,
                "xi": xi,
                "kurtosis_sim": round(
                    kurtosis_sim_val,
                    4
                ),
                "error": round(
                    error_total,
                    6
                )
            })

            if error_total < mejor_error:
                mejor_error = error_total

                mejor_params = (
                    v0,
                    theta,
                    xi
                )

print("\nMejores parámetros refinados:")

print(
    f"v0 = {mejor_params[0]}, "
    f"theta = {mejor_params[1]}, "
    f"xi = {mejor_params[2]}"
)

print(
    f"rho = {rho}, "
    f"kappa = {kappa}"
)

print(
    f"Error total = "
    f"{mejor_error:.6f}"
)

print(
    pd.DataFrame(resultados2)
    .sort_values("error")
    .head(5)
    .to_string(index=False)
)


# =============================================================
# 7. SIMULACIÓN FINAL DEL MODELO DE HESTON
# =============================================================

v0_final = mejor_params[0]
theta_final = mejor_params[1]
xi_final = mejor_params[2]
rho_final = rho
kappa_final = kappa

print(
    "\nSimulando el modelo de Heston "
    "con los parámetros calibrados..."
)

log_ret_heston = simular_heston(
    mu=mu_anual,
    v0=v0_final,
    kappa=kappa_final,
    theta=theta_final,
    xi=xi_final,
    rho=rho_final,
    N_sim=1000,
    N_dias=N_dias,
    dt=dt
)

todos_heston = log_ret_heston.flatten()

print("\n" + "=" * 75)
print("COMPARACIÓN FINAL: DATOS REALES, GBM Y HESTON")
print("=" * 75)

print(
    f"{'Estadístico':<25}"
    f"{'Real':>12}"
    f"{'GBM':>16}"
    f"{'Heston':>16}"
)

print("-" * 75)

print(
    f"{'Media diaria':<25}"
    f"{log_ret.mean():>12.6f}"
    f"{todos_sim.mean():>16.6f}"
    f"{np.mean(todos_heston):>16.6f}"
)

print(
    f"{'Volatilidad diaria':<25}"
    f"{log_ret.std():>12.6f}"
    f"{todos_sim.std():>16.6f}"
    f"{np.std(todos_heston):>16.6f}"
)

print(
    f"{'Skewness':<25}"
    f"{log_ret.skew():>12.4f}"
    f"{pd.Series(todos_sim).skew():>16.4f}"
    f"{pd.Series(todos_heston).skew():>16.4f}"
)

print(
    f"{'Kurtosis (exceso)':<25}"
    f"{log_ret.kurt():>12.4f}"
    f"{pd.Series(todos_sim).kurt():>16.4f}"
    f"{pd.Series(todos_heston).kurt():>16.4f}"
)

print(
    f"{'Mínimo':<25}"
    f"{log_ret.min():>12.4f}"
    f"{todos_sim.min():>16.4f}"
    f"{np.min(todos_heston):>16.4f}"
)

print(
    f"{'Máximo':<25}"
    f"{log_ret.max():>12.4f}"
    f"{todos_sim.max():>16.4f}"
    f"{np.max(todos_heston):>16.4f}"
)

print("=" * 75)


# =============================================================
# 8. COMPARACIÓN GRÁFICA FINAL
# =============================================================

fig, ax = plt.subplots(figsize=(12, 6))

x_min = -0.08
x_max = 0.08

x = np.linspace(
    x_min,
    x_max,
    300
)

ax.hist(
    log_ret,
    bins=100,
    density=True,
    color="steelblue",
    alpha=0.5,
    label="Rendimientos reales S&P 500"
)

ax.hist(
    todos_heston,
    bins=100,
    density=True,
    color="green",
    alpha=0.4,
    label="Rendimientos simulados con Heston"
)

curva_normal = stats.norm.pdf(
    x,
    log_ret.mean(),
    log_ret.std()
)

ax.plot(
    x,
    curva_normal,
    color="red",
    linewidth=2.5,
    label="Distribución normal teórica del GBM"
)

ax.text(
    0.02,
    55,
    f"Kurtosis real: {log_ret.kurt():.2f}",
    color="steelblue",
    fontsize=9,
    fontweight="bold"
)

ax.text(
    0.02,
    50,
    (
        "Kurtosis Heston: "
        f"{pd.Series(todos_heston).kurt():.2f}"
    ),
    color="green",
    fontsize=9,
    fontweight="bold"
)

ax.text(
    0.02,
    45,
    (
        "Kurtosis GBM: "
        f"{pd.Series(todos_sim).kurt():.2f}"
    ),
    color="red",
    fontsize=9,
    fontweight="bold"
)

ax.set_title(
    "Comparación de distribuciones: "
    "datos reales, GBM y Heston\n"
    "S&P 500 (2010–2023)",
    fontsize=13,
    fontweight="bold"
)

ax.set_xlabel(
    "Log-rendimiento diario",
    fontsize=11
)

ax.set_ylabel(
    "Densidad",
    fontsize=11
)

ax.set_xlim(
    x_min,
    x_max
)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "comparacion_final.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

