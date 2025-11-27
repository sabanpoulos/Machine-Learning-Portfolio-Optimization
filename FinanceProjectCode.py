import pandas as pd
import os
import numpy as np
import cvxpy as cp
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Step 1: Load all stock data
market_data_path = 'MarketData'
stock_files = [f for f in os.listdir(market_data_path) if f.endswith('.csv')]
print("Files found:", stock_files)

# Initialize a dictionary to hold all stock DataFrames
stock_data = {}
for file in stock_files:
    try:
        symbol = file.replace('.csv', '')
        file_path = os.path.join(market_data_path, file)
        df = pd.read_csv(file_path, index_col=False)
        df['symbol'] = symbol
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 2005]
        stock_data[symbol] = df
    except Exception as e:
        print(f"Error loading {file}: {e}")

# Step 2: Load HICP data
hicp_df = pd.read_csv('HICP-EuroArea.csv')
hicp_df['DATE'] = pd.to_datetime(hicp_df['DATE'])
hicp_df = hicp_df[hicp_df['DATE'].dt.year >= 2005]

# Step 3: Compute technical indicators for each stock
for symbol, df in stock_data.items():
    if not df.empty:
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA21'] = df['close'].rolling(window=21).mean()
        df['Momentum'] = df['close'] - df['close'].shift(12)
        stock_data[symbol] = df

# Combine all stock prices into a single DataFrame
symbols = list(stock_data.keys())
prices = pd.DataFrame()
for symbol in symbols:
    df = stock_data.get(symbol)
    if df is not None and not df.empty:
        s = df.set_index('date')['close'].rename(symbol)
        prices = pd.concat([prices, s], axis=1)

prices = prices.dropna(axis=0, how='any')  # Align dates
prices = prices.sort_index()

# Compute daily returns and annualize
daily_returns = prices.pct_change().dropna()
if daily_returns.empty:
    raise ValueError("No data left after computing returns. Check your input data.")

# Function to optimize portfolio using cvxpy with relaxed constraints
def optimize_portfolio_cvxpy(returns, cov_matrix, target_return):
    num_assets = len(returns)
    weights = cp.Variable(num_assets)
    # Add small regularization to the objective to ensure positive definiteness
    reg = 1e-6 * cp.sum_squares(weights)
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(weights, cov_matrix) + reg),
        [cp.sum(weights) == 1,
         weights >= 0]
    )

    # First solve without return constraint to check feasibility
    problem.solve(verbose=False)
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print("Feasibility problem - no solution found")
        return None

    # Now add the return constraint with some tolerance
    problem = cp.Problem(
        cp.Minimize(cp.quad_form(weights, cov_matrix) + reg),
        [cp.sum(weights) == 1,
         returns @ weights >= target_return - 0.01,  # Allow 1% below target
         weights >= 0]
    )

    problem.solve(solver=cp.SCS, verbose=False)  # SCS is more robust for large problems
    if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Optimization failed with status: {problem.status}")
        return None
    return weights.value

# Prepare features and target for ML model
def prepare_ml_data(stock_data, hicp_df):
    hicp_df = hicp_df.rename(columns={'DATE': 'date', 'HICP - Overall index (ICP.M.U2.N.000000.4.ANR)': 'hicp'})
    hicp_df = hicp_df.set_index('date')
    ml_data = {}
    for symbol, df in stock_data.items():
        if df.empty:
            continue
        df = df.set_index('date')
        df = df.join(hicp_df, how='left')
        df['hicp'].ffill(inplace=True)
        df['hicp'].bfill(inplace=True)
        df['target'] = df['close'].pct_change().shift(-1)
        df = df.dropna(subset=['target', 'MA7', 'MA21', 'Momentum', 'hicp'])
        features = df[['MA7', 'MA21', 'Momentum', 'hicp']]
        target = df['target']
        ml_data[symbol] = {'features': features, 'target': target}
    return ml_data

ml_data = prepare_ml_data(stock_data, hicp_df)

# Train GBM for each asset using time-series cross-validation
def train_gbm_time_series(features, target):
    tscv = TimeSeriesSplit(n_splits=5)
    gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')

    features_array = features.to_numpy()
    features_imputed = imputer.fit_transform(features_array)
    for train_idx, test_idx in tscv.split(features_imputed):
        X_train, X_test = features_imputed[train_idx], features_imputed[test_idx]
        y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        gbm.fit(X_train_scaled, y_train)
        y_pred = gbm.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Fold MSE: {mse:.6f}")
    return gbm, scaler, imputer

gbm_models = {}
for symbol, data in ml_data.items():
    if len(data['features']) > 0 and len(data['target']) > 0:
        print(f"Training GBM for {symbol} with {len(data['features'])} samples...")
        gbm, scaler, imputer = train_gbm_time_series(data['features'], data['target'])
        gbm_models[symbol] = {'model': gbm, 'scaler': scaler, 'imputer': imputer}

# Backtest function with rolling windows
def backtest_rolling(daily_returns, window_size=126, use_ml=False, gbm_models=None, ml_data=None):
    results = []
    failed_dates = []

    # Precompute all possible windows to avoid redundant calculations
    all_windows = []
    for i in range(window_size, len(daily_returns)):
        if use_ml:
            # For ML, we'll compute predictions on the fly
            all_windows.append((i, None))
        else:
            # For historical, precompute the means
            train_returns = daily_returns.iloc[i-window_size:i]
            all_windows.append((i, train_returns.mean() * 252))

    for i, mu_data in all_windows:
        test_date = daily_returns.index[i]

        if use_ml:
            mu = {}
            for symbol in daily_returns.columns:
                if symbol in gbm_models:
                    model_info = gbm_models[symbol]
                    features = ml_data[symbol]['features']
                    features_up_to_now = features.iloc[:i]
                    if len(features_up_to_now) == 0:
                        continue
                    last_features = features_up_to_now.iloc[-1:].to_numpy()
                    last_features_imputed = model_info['imputer'].transform(last_features)
                    last_features_scaled = model_info['scaler'].transform(last_features_imputed)
                    mu_ml = model_info['model'].predict(last_features_scaled)[0] * 252
                    # Add bounds to prevent extreme values
                    if -0.5 <= mu_ml <= 0.5:  # Reasonable bounds for annualized returns
                        mu[symbol] = mu_ml
        else:
            mu = mu_data

        if len(mu) == 0:
            print(f"No valid returns available for {test_date}")
            failed_dates.append(test_date)
            continue

        mu = pd.Series(mu)
        symbols_available = mu.index.intersection(daily_returns.columns)
        if len(symbols_available) == 0:
            print(f"No common symbols available for {test_date}")
            failed_dates.append(test_date)
            continue

        mu = mu.loc[symbols_available]
        cov_matrix = daily_returns[symbols_available].cov() * 252

        # Add stronger regularization to ensure positive definiteness
        cov_matrix += 1e-4 * np.eye(len(symbols_available))

        # Use median return as target to be more robust to outliers
        target_return = mu.median()

        weights = optimize_portfolio_cvxpy(mu.to_numpy(), cov_matrix.to_numpy(), target_return)
        if weights is None:
            print(f"Optimization failed for {test_date}")
            failed_dates.append(test_date)
            continue

        # Compute portfolio return for the next day
        next_day_return = daily_returns.iloc[i][symbols_available]
        portfolio_return = (weights * next_day_return).sum()
        results.append({'date': test_date, 'return': portfolio_return, 'weights': weights})

    print(f"Failed optimization on {len(failed_dates)} dates")
    return pd.DataFrame(results)

# Run backtests
print("Running backtest for historical mean-variance...")
historical_results = backtest_rolling(daily_returns, window_size=126, use_ml=False)

print("Running backtest for ML-based strategy...")
ml_results = backtest_rolling(daily_returns, window_size=126, use_ml=True, gbm_models=gbm_models, ml_data=ml_data)

# Compute cumulative returns for both strategies
if not historical_results.empty:
    historical_results['cumulative_return'] = (1 + historical_results['return']).cumprod()
if not ml_results.empty:
    ml_results['cumulative_return'] = (1 + ml_results['return']).cumprod()

# # Plot cumulative returns
# plt.figure(figsize=(14, 7))
# if not historical_results.empty:
#     plt.plot(historical_results['date'], historical_results['cumulative_return'], label='Historical Mean-Variance', alpha=0.6)
# if not ml_results.empty:
#     plt.plot(ml_results['date'], ml_results['cumulative_return'], label='ML-Based Strategy', alpha=0.6)
# plt.title('Cumulative Returns: Historical vs ML-Based Strategy')
# plt.xlabel('Date')
# plt.ylabel('Cumulative Return')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Print performance metrics
def compute_metrics(results, name):
    if results.empty:
        print(f"{name} - No valid results")
        return None, None, None, None
    returns = results['return']
    cumulative_return = (1 + returns).prod()
    annualized_return = (1 + returns.mean()) ** 252 - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility
    print(f"{name} - Annualized Return: {annualized_return:.2%}, Volatility: {annualized_volatility:.2%}, Sharpe Ratio: {sharpe_ratio:.2f}")
    return cumulative_return, annualized_return, annualized_volatility, sharpe_ratio

print("\nPerformance Metrics:")
hist_cum, hist_ret, hist_vol, hist_sharpe = compute_metrics(historical_results, "Historical Mean-Variance")
ml_cum, ml_ret, ml_vol, ml_sharpe = compute_metrics(ml_results, "ML-Based Strategy")

# Print comparison if both strategies have results
if hist_ret is not None and ml_ret is not None:
    print("\nComparison:")
    print(f"ML outperforms historical by {ml_ret - hist_ret:.2%} in annualized return")
    print(f"ML has {'higher' if ml_vol > hist_vol else 'lower'} volatility ({ml_vol:.2%} vs {hist_vol:.2%})")
    print(f"ML has {'higher' if ml_sharpe > hist_sharpe else 'lower'} Sharpe ratio ({ml_sharpe:.2f} vs {hist_sharpe:.2f})")
    print(f"ML strategy is {'better' if ml_sharpe > hist_sharpe else 'worse'} overall")
