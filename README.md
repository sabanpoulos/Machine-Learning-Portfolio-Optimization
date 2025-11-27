# Machine Learning Portfolio Optimization

**Hybrid model** combining classical mean-variance optimization with **ML-based return prediction** (Gradient Boosting Machines) to improve risk-adjusted returns. The project uses historical covariance matrices for stability and ML to forecast expected returns from technical indicators and macroeconomic data (e.g., Eurozone inflation).

### Key Findings (2005–2025 Backtest)
- **Classical mean-variance** outperformed the hybrid ML approach, achieving a **12.04% annualized return** and **Sharpe ratio of 0.67** (the highest among tested strategies).
- The hybrid model (11.09% return, 0.60 Sharpe) showed potential but underperformed due to limited feature predictive power.
- Both strategies beat the CAC 40 benchmark in risk-adjusted terms.

### Why It Matters
While ML integration adds flexibility, this implementation highlights the enduring strength of classical methods. Future work will focus on richer features and model refinement.
