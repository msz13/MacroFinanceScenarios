# Equations from: *Price Dividend Ratio and Long-Run Stock Returns: A Score-Driven State Space Model*
**Delle Monache, Petrella & Venditti (2020/2021)**  
*ECB Working Paper No. 2369 / Journal of Business & Economic Statistics, 39(4), 1054–1065*

---

## 1. General State Space Representation

The baseline state space model (contemporaneous form):

$$y_t = Z_t \alpha_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, H_t) \tag{1a}$$

$$\alpha_t = T_t \alpha_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, Q_t), \quad t = 1, \ldots, n \tag{1b}$$

**Notation:**
- $y_t$: $N \times 1$ vector of observed variables  
- $\alpha_t$: $m \times 1$ vector of state variables  
- $\varepsilon_t$: $N \times 1$ measurement error vector  
- $\eta_t$: $m \times 1$ state disturbance vector  
- $Z_t, T_t, H_t, Q_t$: time-varying system matrices  

**Orthogonality assumption:** $\mathbb{E}(\varepsilon_t \eta_s') = 0$ for all $t, s$

**Initial condition:** $\alpha_0 \sim \mathcal{N}(a_0, P_0)$, uncorrelated with $\varepsilon_t$ and $\eta_t$ for all $t$

---

## 2. Conditional Log-Likelihood

Conditional on $\mathcal{Y}_{t-1} = \{y_{t-1}, \ldots, y_1\}$ and parameter vector $\theta$:

$$y_t \mid \mathcal{Y}_{t-1};\, \theta \sim \mathcal{N}(Z_t a_t,\; F_t)$$

$$\alpha_t \mid \mathcal{Y}_{t-1};\, \theta \sim \mathcal{N}(a_t,\; P_t)$$

The log-likelihood contribution at time $t$:

$$\ell_t = \log p(y_t \mid \mathcal{Y}_{t-1},\, \theta) \;\propto\; -\frac{1}{2}\left[\log|F_t| + v_t' F_t^{-1} v_t\right] \tag{2}$$

---

## 3. Kalman Filter Recursions

The prediction error, its covariance, the filtered state and its mean square error (MSE):

$$v_t = y_t - Z_t a_t \tag{3a}$$

$$F_t = Z_t P_t Z_t' + H_t \tag{3b}$$

$$a_{t|t} = a_t + P_t Z_t' F_t^{-1} v_t \tag{3c}$$

$$P_{t|t} = P_t - P_t Z_t' F_t^{-1} Z_t P_t \tag{3d}$$

$$a_{t+1} = T_{t+1}\, a_{t|t} \tag{3e}$$

$$P_{t+1} = T_{t+1}\, P_{t|t}\, T_{t+1}' + Q_{t+1}, \qquad t = 1, \ldots, n \tag{3f}$$

**Initialization:** $a_1 = T_1 a_0$, $\quad P_1 = T_1 P_0 T_1' + Q_1$

**Interpretation:**
- $a_t = \mathbb{E}(\alpha_t \mid \mathcal{Y}_{t-1}, \theta)$: predictive filter, with MSE $P_t = \mathbb{E}[(a_t - \alpha_t)(a_t - \alpha_t)' \mid \mathcal{Y}_{t-1}, \theta]$
- $a_{t|t} = \mathbb{E}(\alpha_t \mid \mathcal{Y}_t, \theta)$: real-time filter, with MSE $P_{t|t} = \mathbb{E}[(a_{t|t} - \alpha_t)(a_{t|t} - \alpha_t)' \mid \mathcal{Y}_{t-1}, \theta]$

---

## 4. Score-Driven Law of Motion for Time-Varying Parameters

The time-varying elements of the system matrices are collected in the vector $f_t$. Their score-driven dynamics:

$$f_{t+1} = c + A f_t + B s_t, \qquad s_t = S_t \nabla_t, \quad t = 1, \ldots, n \tag{4}$$

where the score (gradient) and its scaling matrix are:

$$\nabla_t = \frac{\partial \ell_t}{\partial f_t} \tag{5a}$$

$$S_t = -\mathbb{E}_t\!\left[\frac{\partial^2 \ell_t}{\partial f_t \partial f_t'}\right]^{-1} = \mathcal{I}_t^{-1} \tag{5b}$$

**Parameters:**
- $c$: intercept vector  
- $A$: persistence (autoregressive) matrix  
- $B$: sensitivity matrix (loading on the score)  
- $s_t$: scaled score with $\mathbb{E}[s_t \mid \mathcal{Y}_{t-1}] = 0$ and $\text{Var}[s_t \mid \mathcal{Y}_{t-1}] = \mathcal{I}_t^{-1}$

> **Special case:** Setting $B = 0$ recovers the constant-parameter state space model.

---

## 5. Score and Information Matrix (Result 1)

**Result 1.** Given the model (1)–(2), the score vector and information matrix are:

$$\nabla_t = \frac{1}{2}\left[\dot{F}_t'\,(F_t \otimes F_t)^{-1}\,\text{vec}(v_t v_t' - F_t) \;-\; 2\dot{V}_t'\,F_t^{-1}\,v_t\right] \tag{6a}$$

$$\mathcal{I}_t = \frac{1}{2}\left[\dot{F}_t'\,(F_t \otimes F_t)^{-1}\,\dot{F}_t \;+\; 2\dot{V}_t'\,F_t^{-1}\,\dot{V}_t\right] \tag{6b}$$

for $t = 1, \ldots, n$, where:
- $\dot{V}_t = \partial v_t / \partial f_t'$: Jacobian of the prediction error w.r.t. $f_t$  
- $\dot{F}_t = \partial \,\text{vec}(F_t) / \partial f_t'$: Jacobian of the prediction error covariance w.r.t. $f_t$  
- $\otimes$: Kronecker product  
- $\text{vec}(\cdot)$: column-vectorisation operator  

> All elements of $\mathcal{I}_t$ are computed using information up to $t-1$; $\nabla_t$ also depends on the current observation $y_t$ via $v_t$.

---

## 6. Jacobian Filter (Result 2)

**Result 2.** The Jacobian counterparts of the Kalman filter recursions are:

$$\dot{V}_t = -\left[(a_t' \otimes I_N)\,\dot{Z}_t + (a_{t-1|t-1}' \otimes Z_t)\,\dot{T}_t\right] \tag{7a}$$

$$\dot{F}_t = 2N_N\,(Z_t P_t \otimes I_N)\,\dot{Z}_t + 2\,(Z_t \otimes Z_t)\,N_m\,(T_t P_{t-1|t-1} \otimes I_m)\,\dot{T}_t + \dot{H}_t + (Z_t \otimes Z_t)\,\dot{Q}_t \tag{7b}$$

where:
- $\dot{Z}_t = \partial\, \text{vec}(Z_t)/\partial f_t'$, $\;\dot{T}_t = \partial\, \text{vec}(T_t)/\partial f_t'$, $\;\dot{H}_t = \partial\, \text{vec}(H_t)/\partial f_t'$, $\;\dot{Q}_t = \partial\, \text{vec}(Q_t)/\partial f_t'$
- $N_m$: symmetrizer matrix such that $N_n\, \text{vec}(S) = \text{vec}\!\left[\tfrac{1}{2}(S + S')\right]$ for any $n \times n$ matrix $S$

---

## 7. Complete Algorithm

**Initialize:** $a_{0|0},\; a_1,\; P_{0|0},\; P_1,\; f_1$

**For** $t = 1, \ldots, n$:

1. Evaluate system matrices and their Jacobians: $Z_t,\, T_t,\, H_t,\, Q_t,\, \dot{Z}_t,\, \dot{T}_t,\, \dot{H}_t,\, \dot{Q}_t$
2. Compute prediction error and Jacobians: $v_t,\, F_t,\, \dot{V}_t,\, \dot{F}_t$ &nbsp; [using (3a), (3b), (7)]
3. Compute filtered state, score, information matrix: $a_{t|t},\, P_{t|t},\, \nabla_t,\, \mathcal{I}_t,\, s_t$ &nbsp; [using (3c), (3d), (6)]
4. Update time-varying parameters: $f_{t+1}$ &nbsp; [using (4)]
5. Evaluate updated system matrices: $Z_{t+1},\, T_{t+1},\, H_{t+1},\, Q_{t+1}$
6. Predict next state: $a_{t+1},\, P_{t+1}$ &nbsp; [using (3e), (3f)]

**ML estimation:** $\hat{\theta} = \arg\max \sum_{t=1}^{n} \ell_t(\theta)$

---

## 8. Campbell–Shiller Present-Value Decomposition

The log-linearised present value identity (Campbell & Shiller, 1988):

$$pd_t \approx \frac{\rho}{1-\rho}\,\mu_t - \frac{1}{1-\rho}\,g_t + \sum_{j=0}^{\infty} \rho^j \left[\mathbb{E}_t(\Delta d_{t+1+j}) - \mathbb{E}_t(r_{t+1+j})\right] \tag{8}$$

where $pd_t = \log(P_t/D_t)$ is the log price-dividend ratio, $\rho \approx 0.964$ is a linearisation constant, $r_t$ is the log stock return, and $\Delta d_t$ is log dividend growth.

---

## 9. Unobserved Components Model for the Empirical Application

### 9a. Expected return and dividend growth decomposition

$$\mu_t = \bar{\mu}_t + \tilde{\mu}_t \tag{9}$$

$$g_t = \bar{g}_t + \tilde{g}_t \tag{10}$$

where $\bar{\mu}_t$ and $\bar{g}_t$ are slow-moving (permanent) components, and $\tilde{\mu}_t$, $\tilde{g}_t$ are transitory components.

### 9b. Measurement equations

$$r_{t+1} - \mathbb{E}_t(r_{t+1}) = \varepsilon_{\mu,t+1} \tag{21}$$

$$\Delta d_{t+1} - \mathbb{E}_t(\Delta d_{t+1}) = \varepsilon_{d,t+1} \tag{22}$$

$$pd_t = \frac{\rho}{1-\rho}\,\bar{\mu}_t - \frac{1}{1-\rho}\,\bar{g}_t + \tilde{g}_{t-1} - \tilde{\mu}_t + \varepsilon_{pd,t} \tag{23}$$

### 9c. Transition equations

$$\tilde{g}_t = \tilde{g}_{t-1} + \varepsilon_{g,t} \tag{19}$$

$$\tilde{\mu}_t = \rho_\mu\,\tilde{\mu}_{t-1} + \varepsilon_{\mu,t} \tag{20}$$

---

## 10. State Space Representation of the Empirical Model

**Observed variables vector:**

$$y_t = (r_t,\; \Delta d_t,\; pd_t)' \tag{24}$$

**State vector:**

$$\alpha_t = (1,\; \tilde{g}_t,\; \tilde{\mu}_t,\; \tilde{g}_{t-1},\; \varepsilon_{d,t},\; \varepsilon_{g,t},\; \varepsilon_{\mu,t})' \tag{25}$$

**Measurement matrix** $Z_t$ (time-varying due to score-driven $\bar{\mu}_t$, $\bar{g}_t$):

$$Z_t = \begin{pmatrix}
0 & 0 & -1 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 1 & 1 & 0 \\
\frac{\rho}{1-\rho}\bar{\mu}_t - \frac{1}{1-\rho}\bar{g}_t & -\frac{1}{1-\rho} & -1 & 1 & 0 & 0 & 0
\end{pmatrix}$$

**Transition matrix** $T_t$:

$$T_t = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & \rho_\mu & 0 & 0 & 0 & 1 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}$$

---

## 11. Score-Driven Volatility: Time-Varying $Q_t$

The variance-covariance matrix of shocks evolves as:

$$Q_t = \Lambda_t \bar{Q} \Lambda_t' \tag{26}$$

where $\Lambda_t = \text{diag}(\bar{\sigma}_{d,t},\; \bar{\sigma}_{g,t},\; \bar{\sigma}_{\mu,t})$ contains time-varying standard deviations, and $\bar{Q}$ encodes the time-varying correlations $\bar{\rho}_{d,\mu,t}$ and $\bar{\rho}_{g,\mu,t}$.

### Score-driven dynamics for $f_t = (\bar{g}_t,\; \bar{\mu}_t,\; \bar{\sigma}_{d,t},\; \bar{\sigma}_{g,t},\; \bar{\sigma}_{\mu,t},\; \bar{\rho}_{d,\mu,t},\; \bar{\rho}_{g,\mu,t})'$:

$$f_{t+1} = A f_t + B s_t \tag{27}$$

where $A$ and $B$ are **diagonal** matrices. The first two elements of $f_t$ (the steady states $\bar{g}_t$ and $\bar{\mu}_t$) are modelled as **martingales**, so the corresponding diagonal entries of $A$ are zero:

$$A = \text{diag}(0,\; 0,\; a_3,\; a_4,\; a_5,\; a_6,\; a_7)$$

$$B = \text{diag}(b_1,\; b_2,\; b_3,\; b_4,\; b_5,\; b_6,\; b_7)$$

---

## 12. Smoothed Hessian Scaling

To avoid numerical instability in the scaling matrix, the authors use a smoothed estimator:

$$\tilde{S}_t = (1 - \kappa_h)\,S_t + \kappa_h\,\tilde{S}_{t-1} \tag{28}$$

where $\kappa_h \in [0,1]$ is the smoothing coefficient applied to the Hessian term.

---

## Appendix: Kalman Filter in Predictive Filter Form (Appendix A.2)

Expressing all recursions in terms of the predictive filter only:

$$v_t = y_t - Z_t a_t$$

$$F_t = Z_t P_t Z_t' + H_t$$

$$K_t = T_{t+1} P_t Z_t' F_t^{-1} \quad \text{(Kalman gain)}$$

$$a_{t+1} = T_{t+1} a_t + K_t v_t$$

$$P_{t+1} = T_{t+1} P_t T_{t+1}' + Q_{t+1} - K_t F_t K_t'$$

---

## Appendix: Jacobian Recursions in Predictive Filter Form (Appendix A.2, eqs. A.10–A.15)

The Jacobian filter can also be expressed using only the predictive filter. Define:

$$\dot{a}_t = \frac{\partial a_t}{\partial f_t'}, \qquad \dot{P}_t = \frac{\partial \,\text{vec}(P_t)}{\partial f_t'}$$

Then:

$$\dot{V}_t = -\left[(a_t' \otimes I_N)\,\dot{Z}_t + (I_N \otimes Z_t)\,\dot{a}_t\right]$$

$$\dot{a}_{t+1} = (I_m \otimes T_{t+1})\,\dot{a}_t + (a_t' \otimes I_m)\,\dot{T}_{t+1} + \dot{K}_t\,v_t + K_t\,\dot{V}_t$$

$$\dot{P}_{t+1} = (T_{t+1} \otimes T_{t+1})\,\dot{P}_t\,\cdot(\text{symmetrized}) + \dot{Q}_{t+1} - \frac{\partial\,(K_t F_t K_t')}{\partial f_t'}$$

---

## Appendix: Forward Form of the State Space Model (Appendix C.1)

In the forward form (Durbin & Koopman, 2012), the state space is written as:

$$y_t = Z_t^* \alpha_t^* + \varepsilon_t^*$$

$$\alpha_{t+1}^* = T_t^*\,\alpha_t^* + \eta_t^*$$

The time indexing of $T$ and $Q$ shifts by one period relative to the contemporaneous form (1). The score-driven recursions adapt accordingly so that $f_{t+1}$ depends on matrices evaluated at $t$ rather than $t+1$.

---

*Equations numbered as in the ECB Working Paper No. 2369 (2020) / JBES (2021) version.*
