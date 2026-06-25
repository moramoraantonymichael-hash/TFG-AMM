# !pip install yfinance

# ── CELDA 2: Librerías ───────────────────────────────────────
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf

%matplotlib inline

# Semilla para reproducibilidad, además esto sirve para poder fijar los números. 
# Cada vez que generemos números aleatorios, saldrán los mismos
np.random.seed(42)
# BLOQUE 2: ANÁLISIS ESTADÍSTICO DESCRIPTIVO
# =============================================================

# ── CELDA 3: Descarga de datos ───────────────────────────────
ticker = "^GSPC"
sp500  = yf.download(ticker, start="2010-01-01", end="2023-12-31")
print(sp500.head())
print(f"\nDimensiones: {sp500.shape}")
# ── CELDA 4: Extraer precio de cierre ────────────────────────
# Manejo automático del MultiIndex según versión de yfinance
if isinstance(sp500.columns, pd.MultiIndex):
    precio = sp500["Close"][ticker].dropna()
else:
    precio = sp500["Close"].dropna()

precio = precio.astype(float)
precio.name = "SP500"

print(precio.head())
print(f"\nDesde: {precio.index[0].date()}  Hasta: {precio.index[-1].date()}")
print(f"Total sesiones: {len(precio)}")
# ── CELDA 5: Log-rendimientos ────────────────────────────────
log_ret = np.log(precio / precio.shift(1)).dropna()
log_ret.name = "log_rendimientos"

print(log_ret.head(10))
print(f"\nTotal rendimientos: {len(log_ret)}")

print(f"Peor día:  {log_ret.idxmin().date()} → {log_ret.min():.4f} ({log_ret.min()*100:.2f}%)")
print(f"Mejor día: {log_ret.idxmax().date()} → {log_ret.max():.4f} ({log_ret.max()*100:.2f}%)")
# ── CELDA 6: Tabla de estadísticos (2.1) ─────────────────────
media_diaria      = log_ret.mean()
volatilidad_dia   = log_ret.std()
volatilidad_anual = volatilidad_dia * np.sqrt(252)
skewness          = log_ret.skew()
kurtosis          = log_ret.kurt()
minimo            = log_ret.min()
maximo            = log_ret.max()

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
        "~0.04% diario → crecimiento lento y sostenido",
        "Dispersión típica de un día",
        "Riesgo anual ~ 15-20% esperado",
        "Negativo → más caídas extremas que subidas",
        "> 0 → colas más gordas que la normal (GBM falla)",
        "Crash COVID marzo 2020",
        "Rebote COVID marzo 2020"
    ]
})
print(tabla.to_string(index=False))
# ── CELDA 7: Histograma con normal superpuesta (2.2) ─────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(log_ret, bins=100, density=True,
        color="steelblue", alpha=0.6, label="Rendimientos reales S&P 500")
x = np.linspace(log_ret.min(), log_ret.max(), 300)
curva_normal = stats.norm.pdf(x, media_diaria, volatilidad_dia)
ax.plot(x, curva_normal, color="red", linewidth=2,
        label=f"Normal teórica (GBM)\nμ={media_diaria:.5f}, σ={volatilidad_dia:.5f}")
ax.set_title("Distribución de log-rendimientos diarios del S&P 500 (2010–2023)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("histograma_rendimientos.png", dpi=150)
plt.show()
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(log_ret, bins=100, density=True,
        color="steelblue", alpha=0.6, label="Rendimientos reales S&P 500")
x = np.linspace(log_ret.min(), log_ret.max(), 300)
curva_normal = stats.norm.pdf(x, media_diaria, volatilidad_dia)
ax.plot(x, curva_normal, color="red", linewidth=2,
        label=f"Normal teórica (GBM)\nμ={media_diaria:.5f}, σ={volatilidad_dia:.5f}")
ax.set_title("Distribución de log-rendimientos diarios del S&P 500 (2010–2023)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.03, 0.03
           )
plt.tight_layout()
plt.savefig("histograma_rendimientos.png", dpi=150)
plt.show()
# ── Cola izquierda con escala log ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(log_ret, bins=100, density=True,
        color="steelblue", alpha=0.6, label="Datos reales")

x = np.linspace(log_ret.min(), log_ret.max(), 300)
curva_normal = stats.norm.pdf(x, media_diaria, volatilidad_dia)

ax.plot(x, curva_normal, color="darkred", linewidth=2,
        label="Normal (GBM)")

ax.set_xlim(-0.05, 0.00)

ax.set_yscale("log")

ax.set_ylim(1e-3, 1e2)

ax.set_title("Cola izquierda (escala log)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("cola_izquierda_log.png", dpi=150)
plt.show()

# ── CELDA 8: QQ-plot (2.3) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
(osm, osr), (slope, intercept, r) = stats.probplot(log_ret, dist="norm")
ax.scatter(osm, osr, color="steelblue", alpha=0.4, s=8, label="Datos reales")
x_line = np.array([osm[0], osm[-1]])
ax.plot(x_line, slope * x_line + intercept,
        color="red", linewidth=2, label="Normal teórica (GBM)")

ax.set_title("QQ-plot: Log-rendimientos S&P 500 vs Normal (GBM)\n2010–2023",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Cuantiles teóricos (Normal)", fontsize=11)
ax.set_ylabel("Cuantiles empíricos (datos reales)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("qqplot.png", dpi=150)
plt.show()
# ── CELDA 9: Test de Jarque-Bera (2.4) ──────────────────────
jb_stat, jb_pvalue = jarque_bera(log_ret)
print("=" * 45)
print("       TEST DE JARQUE-BERA")
print("=" * 45)
print(f"  H0: los rendimientos son normales")
print(f"  H1: los rendimientos NO son normales")
print("-" * 45)
print(f"  Estadístico JB : {jb_stat:,.2f}")
print(f"  P-valor        : {jb_pvalue:.2e}")
print("-" * 45)
if jb_pvalue < 0.05:
    print("  CONCLUSIÓN: Rechazamos H0 al 95%")
    print("  Los rendimientos NO son normales.")
    print("  El GBM queda formalmente refutado.")
print("=" * 45)
# ── ACF de r_t y r_t² en una sola imagen ───────────────────────

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
    vlines_kwargs={"colors": "steelblue"}
)

axes[0].set_title(
    "ACF de rendimientos $r_t$ — S&P 500 (2010–2023)\n"
    "Autocorrelación generalmente reducida, con algunas excepciones significativas",
    fontsize=12,
    fontweight="bold"
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
    vlines_kwargs={"colors": "darkorange"}
)

axes[1].set_title(
    "ACF de rendimientos al cuadrado $r_t^2$ — S&P 500 (2010–2023)\n"
    "Evidencia de clustering de volatilidad",
    fontsize=12,
    fontweight="bold"
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
    format="png"
)

plt.show()
# =============================================================
# BLOQUE 3: ESTIMACIÓN DE PARÁMETROS DEL GBM
# =============================================================

# ── CELDA 11: Estimación de μ y σ ────────────────────────────
mu_diario      = log_ret.mean()
sigma_diario   = log_ret.std()
mu_anual       = mu_diario * 252
sigma_anual    = sigma_diario * np.sqrt(252)
n              = len(log_ret)
error_estandar = sigma_diario / np.sqrt(n)
z_95           = 1.96
ic_inf         = (mu_diario - z_95 * error_estandar) * 252
ic_sup         = (mu_diario + z_95 * error_estandar) * 252

print("=" * 50)
print("   BLOQUE 3 — Parámetros estimados del GBM")
print("=" * 50)
print(f"  Observaciones        : {n}")
print("-" * 50)
print(f"  μ diario             : {mu_diario:.6f}")
print(f"  μ anualizado         : {mu_anual:.4f}  ({mu_anual*100:.2f}%)")
print(f"  IC 95% μ anual       : [{ic_inf:.4f}, {ic_sup:.4f}]")
print("-" * 50)
print(f"  σ diario             : {sigma_diario:.6f}")
print(f"  σ anualizado         : {sigma_anual:.4f}  ({sigma_anual*100:.2f}%)")
print("=" * 50)

# ── CELDA 12: Simulación GBM ─────────────────────────────────
np.random.seed(42)
N_sim  = 10000
N_dias = len(log_ret)
dt     = 1 / 252
S0     = float(precio.iloc[1])

Z            = np.random.standard_normal(size=(N_sim, N_dias))
incrementos  = np.exp(
    (mu_anual) * dt
    + sigma_anual * np.sqrt(dt) * Z
)
simulaciones = S0 * np.cumprod(incrementos, axis=1)

print(f"Matriz de simulaciones: {simulaciones.shape}")
print(f"Precio inicial S0     : {S0:.2f}")
print(f"Precio final real     : {precio.iloc[-1]:.2f}")
print(f"Precio final simulado (media): {simulaciones[:, -1].mean():.2f}")
# ── CELDA 13: Gráfico de trayectorias (4.1) ──────────────────
np.random.seed(42)
precio_sim = precio.iloc[1:]
fig, ax    = plt.subplots(figsize=(12, 6))
for i in range(20):
    ax.plot(precio_sim.index, simulaciones[i],
            color="gray", alpha=0.4, linewidth=0.8)
ax.plot(precio_sim.index, precio_sim,
        color="steelblue", linewidth=2, label="Precio real S&P 500")
media_tray = simulaciones.mean(axis=0)
ax.plot(precio_sim.index, media_tray,
        color="red", linewidth=1.5, linestyle="--",
        label="Media 10.000 simulaciones GBM")
ax.set_title("Simulación Monte Carlo GBM vs Precio Real\nS&P 500 (2010–2023)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Fecha", fontsize=11)
ax.set_ylabel("Precio ($)", fontsize=11)
ax.set_ylim(0, 10000)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("trayectorias_gbm.png", dpi=150)
plt.show()
# ── CELDA 14: Estadísticos simulados vs reales (4.2) ─────────
np.random.seed(42)
log_ret_sim = np.diff(np.log(simulaciones), axis=1)
todos_sim   = log_ret_sim.flatten()

print("=" * 60)
print("   BLOQUE 4.2 — Estadísticos simulados vs reales")
print("=" * 60)
print(f"{'Estadístico':<25} {'Real':>10} {'GBM simulado':>15}")
print("-" * 60)
print(f"{'Media diaria':<25} {log_ret.mean():>10.6f} {todos_sim.mean():>15.6f}")
print(f"{'Volatilidad diaria':<25} {log_ret.std():>10.6f} {todos_sim.std():>15.6f}")
print(f"{'Skewness':<25} {log_ret.skew():>10.4f} {pd.Series(todos_sim).skew():>15.4f}")
print(f"{'Kurtosis (exceso)':<25} {log_ret.kurt():>10.4f} {pd.Series(todos_sim).kurt():>15.4f}")
print(f"{'Mínimo':<25} {log_ret.min():>10.4f} {todos_sim.min():>15.4f}")
print(f"{'Máximo':<25} {log_ret.max():>10.4f} {todos_sim.max():>15.4f}")
print("=" * 60)
# ── CELDA 15: Kurtosis por trayectoria ───────────────────────
kurtosis_sim = pd.DataFrame(log_ret_sim).apply(lambda fila: fila.kurt(), axis=1)
print(f"Kurtosis media simulada : {kurtosis_sim.mean():.4f}")
print(f"Kurtosis mínima simulada: {kurtosis_sim.min():.4f}")
print(f"Kurtosis máxima simulada: {kurtosis_sim.max():.4f}")
print(f"Kurtosis real S&P 500   : {kurtosis:.4f}")

# ── CELDA 16: Histograma de kurtosis simulada (4.3) ──────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(kurtosis_sim, bins=40, color="steelblue", alpha=0.7,
        label="Kurtosis de 10.000 simulaciones GBM")
ax.axvline(kurtosis, color="red", linewidth=2.5,
           label=f"Kurtosis real S&P 500 = {kurtosis:.2f}")
ax.axvline(kurtosis_sim.mean(), color="orange", linewidth=1.5,
           linestyle="--", label=f"Media simulada = {kurtosis_sim.mean():.4f}")
ax.set_title("Distribución de la kurtosis simulada por el GBM\n"
             "vs kurtosis real del S&P 500 (2010–2023)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Kurtosis (exceso)", fontsize=11)
ax.set_ylabel("Frecuencia", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kurtosis_simulada.png", dpi=150)
plt.show()
# BLOQUE 5: MODELO DE HESTON
# ── CELDA 17: Simulador de Heston ────────────────────────────
def simular_heston(mu, v0, kappa, theta, xi, rho, N_sim, N_dias, dt):
    """
    Simula N_sim trayectorias del modelo de Heston (Euler-Maruyama).

    Parámetros fijos:  rho, kappa  (elegidos de la literatura)
    Parámetros calibrados: v0, theta, xi  (ajustados por grid search)

    Devuelve:
    ---------
    log_ret_heston : matriz (N_sim, N_dias-1) de log-rendimientos
    """
    v = np.zeros((N_sim, N_dias))
    S = np.zeros((N_sim, N_dias))
    v[:, 0] = v0
    S[:, 0] = 1.0

    for t in range(1, N_dias):
        Z1    = np.random.standard_normal(N_sim)
        Z2    = np.random.standard_normal(N_sim)
        Zv    = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        v_pos = np.maximum(v[:, t-1], 0)
        v[:, t] = (v_pos
                   + kappa * (theta - v_pos) * dt
                   + xi * np.sqrt(v_pos) * np.sqrt(dt) * Zv)
        S[:, t] = S[:, t-1] * np.exp(
            (mu - 0.5 * v_pos) * dt
            + np.sqrt(v_pos) * np.sqrt(dt) * Z1
        )
    return np.diff(np.log(S), axis=1)
# ── CELDA 18: Grid search 1er nivel ──────────────────────────
np.random.seed(42)
rho   = -0.7
kappa = 2.0

media_obj    = log_ret.mean()
varianza_obj = log_ret.var()
kurtosis_obj = log_ret.kurt()

v0_grid    = [0.01, 0.02, 0.03, 0.04, 0.05]
theta_grid = [0.01, 0.02, 0.03, 0.04, 0.05]
xi_grid    = [0.2, 0.4, 0.6, 0.8, 1.0]

mejor_error  = np.inf
mejor_params = None
resultados   = []

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(f"Probando {total} combinaciones...")

for v0 in v0_grid:
    for theta in theta_grid:
        for xi in xi_grid:
            sim   = simular_heston(mu=mu_anual, v0=v0, kappa=kappa,
                                   theta=theta, xi=xi, rho=rho,
                                   N_sim=1000, N_dias=N_dias, dt=dt)
            todos = sim.flatten()
            media_sim        = np.mean(todos)
            varianza_sim     = np.var(todos)
            kurtosis_sim_val = pd.Series(todos).kurt()
            error_total      = (((media_sim - media_obj) / media_obj)**2
                                + ((varianza_sim - varianza_obj) / varianza_obj)**2
                                + ((kurtosis_sim_val - kurtosis_obj) / kurtosis_obj)**2)
            resultados.append({"v0": v0, "theta": theta, "xi": xi,
                                "kurtosis_sim": round(kurtosis_sim_val, 4),
                                "error": round(error_total, 6)})
            if error_total < mejor_error:
                mejor_error  = error_total
                mejor_params = (v0, theta, xi)

print(f"\nMejores parámetros (1er nivel):")
print(f"  v0={mejor_params[0]}, theta={mejor_params[1]}, xi={mejor_params[2]}")
print(f"  Error total = {mejor_error:.6f}")
print(pd.DataFrame(resultados).sort_values("error").head(5).to_string(index=False))
# ── CELDA 19: Grid refinado ───────────────────────────────────

np.random.seed(42)
v0_grid    = [0.03, 0.035, 0.04, 0.045, 0.05]
theta_grid = [0.015, 0.02, 0.025, 0.03]
xi_grid    = [0.7, 0.75, 0.8, 0.85, 0.9]

mejor_error  = np.inf
mejor_params = None
resultados2  = []

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(f"Probando {total} combinaciones (grid refinado)...")

for v0 in v0_grid:
    for theta in theta_grid:
        for xi in xi_grid:
            sim   = simular_heston(mu=mu_anual, v0=v0, kappa=kappa,
                                   theta=theta, xi=xi, rho=rho,
                                   N_sim=1000, N_dias=N_dias, dt=dt)
            todos = sim.flatten()
            media_sim        = np.mean(todos)
            varianza_sim     = np.var(todos)
            kurtosis_sim_val = pd.Series(todos).kurt()
            error_total      = (((media_sim - media_obj) / media_obj)**2
                                + ((varianza_sim - varianza_obj) / varianza_obj)**2
                                + ((kurtosis_sim_val - kurtosis_obj) / kurtosis_obj)**2)
            resultados2.append({"v0": v0, "theta": theta, "xi": xi,
                                 "kurtosis_sim": round(kurtosis_sim_val, 4),
                                 "error": round(error_total, 6)})
            if error_total < mejor_error:
                mejor_error  = error_total
                mejor_params = (v0, theta, xi)

print(f"\nMejores parámetros refinados:")
print(f"  v0={mejor_params[0]}, theta={mejor_params[1]}, xi={mejor_params[2]}")
print(f"  rho={rho} (fijo), kappa={kappa} (fijo)")
print(f"  Error total = {mejor_error:.6f}")
print(pd.DataFrame(resultados2).sort_values("error").head(5).to_string(index=False))
# ── CELDA 20: Simulación final Heston ────────────────────────


np.random.seed(42)
v0_final    = mejor_params[0]
theta_final = mejor_params[1]
xi_final    = mejor_params[2]
rho_final   = rho
kappa_final = kappa
print(mejor_params)
print("Simulando Heston con parámetros calibrados...")
log_ret_heston = simular_heston(
    mu=mu_anual, v0=v0_final, kappa=kappa_final,
    theta=theta_final, xi=xi_final, rho=rho_final,
    N_sim=1000, N_dias=N_dias, dt=dt
)
todos_heston = log_ret_heston.flatten()

print("\n" + "=" * 65)
print("   COMPARACIÓN FINAL: Real vs GBM vs Heston")
print("=" * 65)
print(f"{'Estadístico':<25} {'Real':>10} {'GBM':>12} {'Heston':>12}")
print("-" * 65)
print(f"{'Media diaria':<25} {log_ret.mean():>10.6f} {todos_sim.mean():>12.6f} {np.mean(todos_heston):>12.6f}")
print(f"{'Volatilidad diaria':<25} {log_ret.std():>10.6f} {todos_sim.std():>12.6f} {np.std(todos_heston):>12.6f}")
print(f"{'Skewness':<25} {log_ret.skew():>10.4f} {pd.Series(todos_sim).skew():>12.4f} {pd.Series(todos_heston).skew():>12.4f}")
print(f"{'Kurtosis (exceso)':<25} {log_ret.kurt():>10.4f} {pd.Series(todos_sim).kurt():>12.4f} {pd.Series(todos_heston).kurt():>12.4f}")
print(f"{'Mínimo':<25} {log_ret.min():>10.4f} {todos_sim.min():>12.4f} {np.min(todos_heston):>12.4f}")
print(f"{'Máximo':<25} {log_ret.max():>10.4f} {todos_sim.max():>12.4f} {np.max(todos_heston):>12.4f}")
print("=" * 65)
# ── CELDA 21: Histograma comparativo final ───────────────────
fig, ax  = plt.subplots(figsize=(12, 6))
x_min, x_max = -0.08, 0.08
x = np.linspace(x_min, x_max, 300)

ax.hist(log_ret, bins=100, density=True,
        color="steelblue", alpha=0.5, label="Rendimientos reales S&P 500")
ax.hist(todos_heston, bins=100, density=True,
        color="green", alpha=0.4, label="Heston simulado")
curva_normal = stats.norm.pdf(x, log_ret.mean(), log_ret.std())
ax.plot(x, curva_normal, color="red", linewidth=2.5,
        label="Normal teórica (GBM)  Kurt≈0")

ax.text(0.02, 55, f"Kurtosis real:   {log_ret.kurt():.2f}",
        color="steelblue", fontsize=9, fontweight="bold")
ax.text(0.02, 50, f"Kurtosis Heston: {pd.Series(todos_heston).kurt():.2f}",
        color="green", fontsize=9, fontweight="bold")
ax.text(0.02, 45, f"Kurtosis GBM:    {pd.Series(todos_sim).kurt():.2f}",
        color="red", fontsize=9, fontweight="bold")

ax.set_title("Comparación de distribuciones: Real vs GBM vs Heston\n"
             "S&P 500 (2010–2023)", fontsize=13, fontweight="bold")
ax.set_xlabel("Log-rendimiento diario", fontsize=11)
ax.set_ylabel("Densidad", fontsize=11)
ax.set_xlim(x_min, x_max)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparacion_final.png", dpi=150)
plt.show()

# =============================================================
# 6. CALIBRACIÓN DEL MODELO DE HESTON (con condición de Feller)


np.random.seed(42)
rho   = -0.7
kappa =  2.0

media_obj    = log_ret.mean()
varianza_obj = log_ret.var()
kurtosis_obj = log_ret.kurt()

# --- Grid grueso ---
# Con kappa=2.0, Feller exige theta > xi²/4
# xi ∈ [0.1, 0.3] → theta mínimo razonable ≈ 0.01–0.03
# Subimos theta y bajamos xi respecto al grid anterior

v0_grid    = [0.01, 0.02, 0.03, 0.04, 0.05]
theta_grid = [0.03, 0.05, 0.07, 0.10, 0.15]
xi_grid    = [0.10, 0.15, 0.20, 0.25, 0.30]

mejor_error  = np.inf
mejor_params = None
resultados   = []

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(f"Probando {total} combinaciones en la primera calibración...")

combinaciones_validas = 0

for v0 in v0_grid:
    for theta in theta_grid:
        for xi in xi_grid:

            # --- FILTRO DE FELLER ---
            if 2 * kappa * theta <= xi ** 2:
                continue

            combinaciones_validas += 1

            sim  = simular_heston(
                mu=mu_anual, v0=v0, kappa=kappa,
                theta=theta, xi=xi, rho=rho,
                N_sim=1000, N_dias=N_dias, dt=dt
            )
            todos = sim.flatten()

            media_sim    = np.mean(todos)
            varianza_sim = np.var(todos)
            kurtosis_sim_val = pd.Series(todos).kurt()

            error_media    = ((media_sim    - media_obj)    / media_obj)    ** 2
            error_varianza = ((varianza_sim - varianza_obj) / varianza_obj) ** 2
            error_kurtosis = ((kurtosis_sim_val - kurtosis_obj) / kurtosis_obj) ** 2

            error_total = error_media + error_varianza + error_kurtosis

            resultados.append({
                "v0": v0, "theta": theta, "xi": xi,
                "feller": round(2 * kappa * theta - xi ** 2, 4),
                "kurtosis_sim": round(kurtosis_sim_val, 4),
                "error": round(error_total, 6)
            })

            if error_total < mejor_error:
                mejor_error  = error_total
                mejor_params = (v0, theta, xi)

print(f"Combinaciones que cumplen Feller: {combinaciones_validas}/{total}")
print(f"\nMejores parámetros (1er nivel):")
print(f"  v0={mejor_params[0]}, theta={mejor_params[1]}, xi={mejor_params[2]}")
print(f"  Feller: 2κθ - ξ² = {2*kappa*mejor_params[1] - mejor_params[2]**2:.4f} > 0 ✓")
print(f"  Error total = {mejor_error:.6f}")
print(pd.DataFrame(resultados).sort_values("error").head(5).to_string(index=False))


# --- Grid refinado alrededor de los mejores parámetros ---
v0_best, theta_best, xi_best = mejor_params

v0_grid    = [max(0.005, v0_best    + d) for d in [-0.015, -0.01, 0.0, 0.01, 0.015]]
theta_grid = [max(0.01,  theta_best + d) for d in [-0.02,  -0.01, 0.0, 0.01, 0.02 ]]
xi_grid    = [max(0.05,  xi_best    + d) for d in [-0.05,  -0.02, 0.0, 0.02, 0.05 ]]

mejor_error  = np.inf
mejor_params = None
resultados2  = []

total = len(v0_grid) * len(theta_grid) * len(xi_grid)
print(f"\nProbando {total} combinaciones en el grid refinado...")

combinaciones_validas2 = 0

for v0 in v0_grid:
    for theta in theta_grid:
        for xi in xi_grid:

            # --- FILTRO DE FELLER ---
            if 2 * kappa * theta <= xi ** 2:
                continue

            combinaciones_validas2 += 1

            sim  = simular_heston(
                mu=mu_anual, v0=v0, kappa=kappa,
                theta=theta, xi=xi, rho=rho,
                N_sim=200, N_dias=N_dias, dt=dt
            )
            todos = sim.flatten()

            media_sim        = np.mean(todos)
            varianza_sim     = np.var(todos)
            kurtosis_sim_val = pd.Series(todos).kurt()

            error_media    = ((media_sim    - media_obj)    / media_obj)    ** 2
            error_varianza = ((varianza_sim - varianza_obj) / varianza_obj) ** 2
            error_kurtosis = ((kurtosis_sim_val - kurtosis_obj) / kurtosis_obj) ** 2

            error_total = error_media + error_varianza + error_kurtosis

            resultados2.append({
                "v0": v0, "theta": theta, "xi": xi,
                "feller": round(2 * kappa * theta - xi ** 2, 4),
                "kurtosis_sim": round(kurtosis_sim_val, 4),
                "error": round(error_total, 6)
            })

            if error_total < mejor_error:
                mejor_error  = error_total
                mejor_params = (v0, theta, xi)

print(f"Combinaciones que cumplen Feller: {combinaciones_validas2}/{total}")
print(f"\nMejores parámetros refinados:")
print(f"  v0={mejor_params[0]:.4f}, theta={mejor_params[1]:.4f}, xi={mejor_params[2]:.4f}")
print(f"  rho={rho} (fijo), kappa={kappa} (fijo)")
print(f"  Feller: 2κθ - ξ² = {2*kappa*mejor_params[1] - mejor_params[2]**2:.4f} > 0 ✓")
print(f"  Error total = {mejor_error:.6f}")
print(pd.DataFrame(resultados2).sort_values("error").head(5).to_string(index=False))
