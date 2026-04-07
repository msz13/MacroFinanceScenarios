# Expected Returns with Cash Flow Trends and Cycles

## Analysis of Hillenbrand & McCarthy (2026)

**Authors:** Sebastian Hillenbrand (Harvard Business School) & Odhrain McCarthy (NYU Stern / NYU Abu Dhabi)

**Date:** March 2026

---

## 1. Overview

The paper argues that cash flow growth contains a **permanent trend component**, rendering the price-dividend ratio a noisy proxy for expected returns. By jointly filtering trend growth, cash flow cycles, and expected returns via an **Extended Kalman Filter (EKF)**, the authors show that this cash flow noise generates severe attenuation bias in standard predictive regressions. Purging cash-flow trends and cycles restores return predictability, delivering out-of-sample R² of **9% at one-year** and **22% at five-year** horizons.

---

## 2. Cash Flow Decomposition

Log dividends $d_t$ are decomposed into a **non-stationary trend** $\tau_t$ and a **stationary cycle** $c_t$:

$$
d_t = \tau_t + c_t
$$

### 2.1 Trend Component

The trend evolves as a **local linear trend** (integrated random walk):

$$
\tau_t = \tau_{t-1} + g_{t-1}
$$

$$
g_t = g_{t-1} + \varepsilon_{g,t}, \quad \varepsilon_{g,t} \sim \mathcal{N}(0, \sigma_g^2)
$$

- $g_t$ is the **trend growth rate** — a random walk, so the trend can permanently accelerate or decelerate.
- Setting $\sigma_g^2 = 0$ recovers constant trend growth (classical present-value models).
- The trend level $\tau_t$ evolves deterministically given $g_t$ (no idiosyncratic shock), yielding a smooth integrated random walk trend.

### 2.2 Cycle Component

The cycle follows a stationary **ARMA(1,1)** process:

$$
c_t = \rho_c \, c_{t-1} + \theta_c \, \varepsilon_{c,t-1} + \varepsilon_{c,t}, \quad \varepsilon_{c,t} \sim \mathcal{N}(0, \sigma_c^2)
$$

- $|\rho_c| < 1$ ensures stationarity.
- Captures the mean-reverting cash flow dynamics (dividend smoothing, business-cycle fluctuations).

---

## 3. Expected Returns

Expected returns $E_t[r_{t,t+1}] \equiv \mu_t$ follow an **AR(1)** process around a constant steady-state level $\bar{\mu}$:

$$
\mu_t = \bar{\mu} + \rho_\mu (\mu_{t-1} - \bar{\mu}) + \varepsilon_{\mu,t}, \quad \varepsilon_{\mu,t} \sim \mathcal{N}(0, \sigma_\mu^2)
$$

**All shocks** $\varepsilon_{g,t}$, $\varepsilon_{c,t}$, and $\varepsilon_{\mu,t}$ **are mutually uncorrelated** (identification assumption).

### Model-Implied Expected Returns over $j$ Periods

$$
\hat{E}_t[r_{t,t+j}] = \frac{1 - \rho^j}{1 - \rho} \hat{\bar{\mu}} + \frac{1 - (\rho \hat{\rho}_\mu)^j}{1 - \rho \hat{\rho}_\mu} (\hat{\mu}_t - \hat{\bar{\mu}})
$$

- First term: present value of the long-run mean expected return.
- Second term: cyclical deviation of expected returns from their long-run mean.

### Volatility of Expected Returns

$$
\sigma(E_t[r_{t,t+j}]) = \frac{1 - (\rho \rho_\mu)^j}{1 - \rho \rho_\mu} \cdot \frac{\sigma_\mu}{\sqrt{1 - \rho_\mu^2}}
$$

---

## 4. Valuation Ratio (Price-Dividend Ratio)

### 4.1 Log-Return Definition

$$
r_{t,t+1} = \Delta d_{t+1} + \log(1 + \exp(pd_{t+1})) - pd_t
$$

### 4.2 Steady-State Price-Dividend Ratio

Following Lettau & Van Nieuwerburgh (2008), log-linearization is performed around a **time-varying** steady-state ratio $\overline{pd}_t$ (not a fixed mean), defined for $g_t < \bar{\mu}$:

$$
\overline{pd}_t \equiv pd(g_t) = g_t - \log(\exp(\bar{\mu}) - \exp(g_t))
$$

Higher trend growth $g_t$ increases $\overline{pd}_t$; higher average returns $\bar{\mu}$ decrease it.

### 4.3 Log-Linearized Present-Value Identity

$$
pd_t - \overline{pd}_t = \delta + \sum_{j=1}^{\infty} \rho^{j-1} E_t[\Delta d_{t+j} - g_t] - \sum_{j=1}^{\infty} \rho^{j-1} E_t[r_{t+j-1,t+j} - \bar{\mu}]
$$

where $\delta$ is a Jensen's inequality constant close to zero and $\rho \approx 0.96$.

### 4.4 Valuation Ratio Decomposition

Substituting the cash flow and expected return dynamics:

$$
pd_t = \delta + \overline{pd}_t + M_t + C_t
$$

where the three components are:

**Non-stationary trend component** (permanent cash flow movements):

$$
\overline{pd}_t = g_t - \log(\exp(\bar{\mu}) - \exp(g_t))
$$

**Expected-return cycle** (discount rate variation):

$$
M_t = -\frac{1}{1 - \rho \rho_\mu}(\mu_t - \bar{\mu})
$$

**Cash-flow cycle** (transitory cash flow variation):

$$
C_t = -\frac{1 - \rho_c}{1 - \rho \rho_c} c_t + \frac{(1 - \rho) \theta_c}{1 - \rho \rho_c} \varepsilon_{c,t}
$$

**Jensen's inequality term:**

$$
\delta = \frac{\rho}{2(1-\rho)^2} \sigma_g^2
$$

### 4.5 Interpretation

| Component | Role | Stationarity |
|---|---|---|
| $\overline{pd}_t$ | Permanent shifts in trend growth | Non-stationary (random walk in $g_t$) |
| $M_t$ | Expected-return cycle (discount rates) | Stationary (AR(1) via $\mu_t$) |
| $C_t$ | Cash-flow cycle | Stationary (ARMA(1,1) via $c_t$) |

---

## 5. Attenuation Bias in Predictive Regressions

### Predictive Slope Coefficient

When predicting $j$-period returns $r_{t,t+j}$ with $pd_t$:

$$
b_j = -[1 - (\rho \rho_\mu)^j] \left[1 - \frac{\text{Var}(\eta_t)}{\text{Var}(pd_t)}\right]
$$

where $\eta_t = pd(g_t) + C_t$ is the cash-flow noise component.

Equivalently:

$$
b_j = -[1 - (\rho \rho_\mu)^j] \left[1 - \frac{\text{Var}(pd(g_t)) + \text{Var}(C_t)}{\text{Var}(pd_t)}\right]
$$

- The first factor $-[1 - (\rho\rho_\mu)^j]$ reflects that $pd_t$ contains the signal for the entire term structure of returns (converges to $-1$ as $j \to \infty$).
- The second factor captures **attenuation from cash-flow noise**: as $\text{Var}(\eta_t)$ increases, the predictive coefficient shrinks toward zero.

### Unattenuated Coefficient (using $M_t$ directly)

$$
b_j = -[1 - (\rho_\mu \rho)^j]
$$

---

## 6. Variance Decomposition of $pd_t$

$$
1 = b_{\overline{pd}} + b_M + b_C
$$

where:

$$
b_{\overline{pd}} = \frac{\text{Var}(\overline{pd}_t)}{\text{Var}(pd_t)}, \quad b_M = \frac{\text{Var}(M_t)}{\text{Var}(pd_t)}, \quad b_C = \frac{\text{Var}(C_t)}{\text{Var}(pd_t)}
$$

### Variance of Each Component

**Variance of expected-return component:**

$$
\text{Var}(M_t) = \frac{1}{(1 - \rho \rho_\mu)^2} \cdot \frac{\sigma_\mu^2}{1 - \rho_\mu^2}
$$

**Variance of cash-flow cycle component:**

$$
\text{Var}(C_t) = \sigma_c^2 \left[\beta_c^2 \cdot \frac{1 + \theta_c^2 + 2\rho_c \theta_c}{1 - \rho_c^2} + \beta_\varepsilon^2 + 2\beta_c \beta_\varepsilon \right]
$$

where $\beta_c = -\frac{1-\rho_c}{1-\rho\rho_c}$ and $\beta_\varepsilon = \frac{\theta_c(1-\rho)}{1-\rho\rho_c}$.

**Variance of trend pd component (second-order approximation):**

$$
\text{Var}(pd(g_t)) \approx \frac{\sigma_g^2 \cdot T}{(1-\rho)^2} + \frac{1}{2} \frac{\rho^2}{(1-\rho)^4} \sigma_g^4 \cdot T^2
$$

This variance **grows with sample length** $T$ because trend growth shocks are permanent, while $\text{Var}(M_t)$ and $\text{Var}(C_t)$ are fixed (stationary processes).

### Horizon-Dependent Decomposition (Model-Implied Estimates)

| Horizon $T$ | $b_M$ (Expected Returns) | $b_C$ (Cash-Flow Cycle) | $b_{\overline{pd}}$ (Trend Growth) |
|---|---|---|---|
| 1 year | 0.84 | 0.14 | 0.02 |
| 10 years | 0.72 | 0.12 | 0.16 |
| 50 years | 0.44 | 0.08 | 0.48 |
| 150 years | 0.22 | 0.04 | 0.74 |

**Key insight:** Discount rate variation dominates at short/medium horizons; trend growth dominates at long horizons.

### Decomposition of Stationary Component $pd_t - \overline{pd}_t$

$$
\frac{\text{Var}(M_t)}{\text{Var}(pd_t - \overline{pd}_t)} = 0.85, \quad \frac{\text{Var}(C_t)}{\text{Var}(pd_t - \overline{pd}_t)} = 0.15
$$

(Constant across all horizons.)

---

## 7. State-Space Representation

### 7.1 State Vector

$$
s_t = (\tau_t, \; g_t, \; c_t, \; \varepsilon_{c,t}, \; \mu_t, \; 1)^\top
$$

### 7.2 Transition Equation

$$
s_{t+1} = A \, s_t + C \, \tilde{\varepsilon}_{t+1}
$$

$$
A = \begin{pmatrix}
1 & 1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & \rho_c & \theta_c & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & \rho_\mu & (1-\rho_\mu)\bar{\mu} \\
0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

$$
C = \begin{pmatrix}
0 & 0 & 0 \\
\sigma_g & 0 & 0 \\
0 & \sigma_c & 0 \\
0 & \sigma_c & 0 \\
0 & 0 & \sigma_\mu \\
0 & 0 & 0
\end{pmatrix}
$$

where $\tilde{\varepsilon}_{t+1} = (\tilde{\varepsilon}_{g,t+1}, \; \tilde{\varepsilon}_{c,t+1}, \; \tilde{\varepsilon}_{\mu,t+1})^\top \sim \mathcal{N}(0, I_3)$ are mutually independent standardized innovations.

- $C$ is a $6 \times 3$ shock-loading matrix governing how each structural innovation enters and scales the state variables.
- $A$ determines the loading on past states.

### 7.3 Measurement Equation

Observable vector:

$$
x_t = (d_t, \; pd_t)^\top
$$

Because the steady-state log price-dividend ratio $\overline{pd}_t$ is a **nonlinear function** of $g_t$, the standard linear Kalman filter cannot be applied. The paper uses an **Extended Kalman Filter (EKF)** which replaces the nonlinear measurement function with its first-order Taylor approximation at each time step.

The linearized measurement equation is $x_t = G_t \, s_t$, where:

$$
G_t = \begin{pmatrix}
1 & 0 & 1 & 0 & 0 & 0 \\
0 & \lambda_t & -\dfrac{1 - \rho_c}{1 - \rho\rho_c} & \dfrac{(1-\rho)\theta_c}{1 - \rho\rho_c} & -\dfrac{1}{1 - \rho_\mu \rho} & \kappa_t
\end{pmatrix}
$$

with:

$$
\lambda_t = 1 + \frac{\exp(\hat{g}_{t|t-1})}{\exp(\bar{\mu}) - \exp(\hat{g}_{t|t-1})}
$$

$$
\kappa_t = \delta + \frac{\bar{\mu}}{1 - \rho\rho_\mu} + \hat{g}_{t|t-1} - \log(\exp(\bar{\mu}) - \exp(\hat{g}_{t|t-1})) - \lambda_t \hat{g}_{t|t-1}
$$

- **First row** of $G_t$: maps dividends to the trend and cycle ($d_t = \tau_t + c_t$).
- **Second row**: encodes the structural present-value constraint.
- At each step, $G_t$ is re-evaluated at the current prior estimate $\hat{g}_{t|t-1}$, so the linearization tracks the evolving steady-state of the pd ratio.

---

## 8. Estimated Parameters (Baseline Model)

Estimated at **quarterly frequency** via Maximum Likelihood using the EKF on the sample **1871–2022**.

| Parameter | Symbol | Estimate | Std. Error | Interpretation |
|---|---|---|---|---|
| Cycle AR(1) | $\rho_c$ | 0.9710 | (0.0080) | Quarterly persistence of dividend cycle level |
| Cycle MA(1) | $\theta_c$ | 0.4927 | (0.0823) | MA(1) coefficient of cash-flow cycle |
| Trend growth vol. | $\sigma_g \times 100$ | 0.0226 | (0.0166) | Quarterly vol. of trend growth (≈ 18 bps annualized) |
| Cycle vol. | $\sigma_c$ | 0.0273 | (0.0044) | Quarterly vol. of cash-flow cycle shock |
| Expected return AR(1) | $\rho_\mu$ | 0.9556 | (0.0357) | Quarterly persistence of expected returns |
| Expected return vol. | $\sigma_\mu$ | 0.0048 | (0.0031) | Quarterly vol. of expected return innovations |
| Steady-state return | $\bar{\mu}$ | 0.0215 | (0.0030) | Quarterly steady-state expected return (≈ 8.6% ann.) |

Standard errors are **bootstrapped** (parametric moving-block bootstrap, B=1000, block length L=20 quarters).

### Derived Annual Quantities

| Quantity | Value | Derivation |
|---|---|---|
| Annual trend growth volatility | ≈ 18 bps | $4\sigma_g\sqrt{4}$ |
| Annual cycle AR(1) (dividend level) | $\rho_c^4 \approx 0.89$ | Quarterly-to-annual conversion |
| Annual cycle growth AR(1) | ≈ −0.11 | Approximately i.i.d. but slowly mean-reverting |
| Annual cycle growth volatility | ≈ 7.5% | From ARMA(1,1) autocovariance structure |
| Annual expected return AR(1) | $\rho_\mu^4 \approx 0.83$ | Half-life ≈ 4 years |
| Annual expected return volatility | ≈ 6.1% | $\sigma_\mu / \sqrt{1-\rho_\mu^2}$ annualized |
| Linearization constant | $\rho \approx 0.96$ | Annual; quarterly $\rho \approx 0.99$ |

---

## 9. Persistence of the Valuation Ratio

The AR(1) coefficient of $pd_t$ is a **variance-weighted average** of its components' persistences:

$$
\phi_{pd} = \phi_{\overline{pd}} \cdot b_{\overline{pd}} + \phi_M \cdot b_M + \phi_C \cdot b_C
$$

| Horizon $T$ | $\phi_{pd}$ (overall) | $\phi_M$ (expected returns) | $\phi_C$ (cash-flow cycle) | $\phi_{\overline{pd}}$ (trend) |
|---|---|---|---|---|
| 1 year | 0.85 | 0.83 | 0.90 | 1.00 |
| 10 years | 0.87 | 0.83 | 0.90 | 1.00 |
| 50 years | 0.92 | 0.83 | 0.90 | 1.00 |
| 150 years | 0.96 | 0.83 | 0.90 | 1.00 |

The high observed persistence of $pd_t$ does not reflect slow-moving discount rates; it is inherited from the permanent trend-growth component.

---

## 10. Out-of-Sample Return Predictability

| Metric | Raw Returns 1-yr | Raw Returns 5-yr | Excess Returns 1-yr | Excess Returns 5-yr |
|---|---|---|---|---|
| **In-Sample R² (full sample)** | 2.37% | 15.31% | 3.87% | 21.24% |
| **In-Sample R² (post-1965)** | 7.65% | 19.86% | 8.13% | 17.22% |
| **Out-of-Sample R²** (post-1965) | **8.90%\*\*** | **22.38%\*\*** | **9.12%\*\*** | **16.78%\*\*** |

Traditional valuation ratios (PD, PE, CAPE) all produce **negative** out-of-sample R².

---

## 11. Extension: Drifting Steady-State Expected Returns

Replacing the constant $\bar{\mu}$ with a random walk:

$$
\mu_t = \bar{\mu}_t + \rho_\mu(\mu_{t-1} - \bar{\mu}_{t-1}) + \varepsilon_{\mu,t}
$$

$$
\bar{\mu}_t = \bar{\mu}_{t-1} + \varepsilon_{\bar{\mu},t}, \quad \varepsilon_{\bar{\mu},t} \sim \mathcal{N}(0, \sigma_{\bar{\mu}}^2)
$$

### Extended State Vector

$$
s_t = (\tau_t, \; g_t, \; c_t, \; \varepsilon_{c,t}, \; \bar{\mu}_t, \; \mu_t, \; 1)^\top
$$

### Extended Transition and Shock-Loading Matrices

$$
A = \begin{pmatrix}
1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & \rho_c & \theta_c & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1-\rho_\mu & \rho_\mu & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

$$
C = \begin{pmatrix}
0 & 0 & 0 & 0 \\
\sigma_g & 0 & 0 & 0 \\
0 & \sigma_c & 0 & 0 \\
0 & \sigma_c & 0 & 0 \\
0 & 0 & \sigma_{\bar{\mu}} & 0 \\
0 & 0 & \sigma_{\bar{\mu}} & \sigma_\mu \\
0 & 0 & 0 & 0
\end{pmatrix}
$$

Key finding: trend growth is largely unchanged across specifications; if anything, steady-state expected returns have **increased** from roughly 7% p.a. pre-1900 to 8–8.5% by the 1980s.

---

## 12. Key Conclusions

1. **Cash flows contain a permanent trend component** that makes the price-dividend ratio a noisy proxy for expected returns.
2. **Attenuation bias** in standard predictive regressions is severe — the univariate slope is attenuated by roughly 5–6× at the one-year horizon.
3. **Purging cash-flow noise restores return predictability**: out-of-sample R² of 9% (1-year) and 22% (5-year).
4. **Horizon-dependent decomposition**: discount rates dominate price-dividend variation at short/medium horizons (84% at 1 year), while trend growth dominates at long horizons (74% at 150 years).
5. The **high persistence** of valuation ratios reflects the permanent trend-growth component, not slow-moving discount rates.
6. Results are **robust** across dividend vs. earnings specifications and constant vs. drifting steady-state expected returns.

---

*Analysis based on: Hillenbrand, Sebastian and McCarthy, Odhrain, "Expected Returns with Cash Flow Trends and Cycles" (March 2026). Working Paper.*
