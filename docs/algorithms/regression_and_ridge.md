# Unit 2: OLS and Ridge Regression from Scratch

## 1. What the Algorithms Do

Ordinary Least Squares (OLS) and Ridge Regression model the continuous physical relationship between ambient conditions, operating modes, and total rooftop unit electrical power consumption (`RTU_TOT_WATT`).

In HVAC FaultSense, regression serves as a **healthy baseline energy estimator**:
$$\hat{y}_t = \mathbf{x}_t \mathbf{w} + b$$
The residual represents unmodeled deviation:
$$e_t = y_t - \hat{y}_t$$

Persistent, large positive or negative residuals indicate that the machine is operating away from its expected energy baseline, providing **candidate evidence for anomalous behavior**.

## 2. Mathematical Foundation & Equations

### A. Ordinary Least Squares (OLS)
Mean Squared Error (MSE) objective:
$$J_{\text{OLS}}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\mathbf{x}_i \mathbf{w} + b))^2$$

Gradients with respect to weights and bias:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

Gradient Descent updates:
$$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$
$$b^{(k+1)} = b^{(k)} - \alpha \frac{\partial J}{\partial b}$$

### B. Ridge Regression (L2 Regularization)
Ridge adds an L2 weight shrinkage penalty to handle multicollinearity among collinear temperature and pressure sensors:
$$J_{\text{Ridge}}(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (\mathbf{x}_i \mathbf{w} + b))^2 + \lambda \sum_{j=1}^{p} w_j^2$$

**Important:** The intercept $b$ is strictly unpenalized.

Weight gradient with L2 shrinkage:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) + 2 \lambda \mathbf{w}$$
$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)$$

Analytical normal equation with centered features:
$$\mathbf{w}_{\text{Ridge}} = (\mathbf{X}_c^T \mathbf{X}_c + n \lambda \mathbf{I})^{-1} \mathbf{X}_c^T \mathbf{y}_c, \quad b = \bar{y} - \bar{\mathbf{X}} \mathbf{w}$$

## 3. Implementation Workflow

1. Standardize features $z = (X - \mu_{\text{train}}) / \sigma_{\text{train}}$.
2. Initialize weights $\mathbf{w} \sim \mathcal{N}(0, 0.01)$ and bias $b = \bar{y}$.
3. Iteratively compute forward predictions, residuals, MSE loss, and analytical gradients.
4. Update parameters using learning rate $\alpha = 0.02$.
5. Terminate when $|J^{(k)} - J^{(k-1)}| < 10^{-6}$ or at $k = 1500$ iterations.

## 4. Evaluation Metrics & Measured Results

| Model | Task | MAE (W) | RMSE (W) | $R^2$ |
|---|---|---|---|---|
| **Mean Baseline** | Regression | 794.23 | 944.59 | -0.0056 |
| **OLS Regression (Scratch)** | Regression | **29.92** | **38.98** | **0.9983** |
| **Ridge Regression (Scratch)** | Regression | 137.91 | 169.09 | 0.9678 |
| **Linear Regression (Reference)** | Regression | 0.00 | 0.00 | 1.0000 |

## 5. Strengths & Limitations

- **Strengths:** Highly explainable, fast convergence, mathematically grounded, coefficients map directly to physical subsystem contributions.
- **Limitations:** Cannot directly categorize the root cause of a fault; an elevated residual signals anomalous power draw, but not whether it is caused by condenser fouling or a refrigerant leak.

