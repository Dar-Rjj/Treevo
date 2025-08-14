import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n_days=10, m_days=20, p_days=10, q_days=30, sector_col='sector', macro_data=None):
    # Obtain Close Prices
    close_prices = df['close']
    
    # Calculate Log Returns
    log_returns = np.log(close_prices).diff()
    
    # Compute Momentum
    cum_log_returns = log_returns.rolling(window=n_days).sum()
    upper_threshold = cum_log_returns.mean() + 2 * cum_log_returns.std()
    lower_threshold = cum_log_returns.mean() - 2 * cum_log_returns.std()
    momentum = np.clip(cum_log_returns, a_min=lower_threshold, a_max=upper_threshold)
    
    # Adjust for Volume
    volumes = df['volume']
    mean_volume = volumes.rolling(window=n_days).mean()
    volume_adjusted_momentum = momentum * (volumes / mean_volume)
    
    # Determine Absolute Price Changes
    abs_price_changes = close_prices.diff().abs()
    
    # Calculate Advanced Volatility Measures
    std_abs_price_changes = abs_price_changes.rolling(window=m_days).std()
    ema_abs_price_changes = abs_price_changes.ewm(span=p_days, adjust=False).mean()
    iqr_abs_price_changes = abs_price_changes.rolling(window=q_days).quantile(0.75) - abs_price_changes.rolling(window=q_days).quantile(0.25)
    
    # Incorporate Sector-Specific Indicators
    sectors = df[sector_col]
    unique_sectors = sectors.unique()
    sector_returns = {s: close_prices[sectors == s].pct_change().mean() for s in unique_sectors}
    sector_relative_strength = sectors.map(sector_returns)
    
    # Incorporate Macroeconomic Indicators
    if macro_data is not None:
        interest_rate_changes = macro_data['interest_rate'].diff()
        gdp_growth = macro_data['gdp_growth']
        macro_factors = interest_rate_changes + gdp_growth
    else:
        macro_factors = pd.Series(index=df.index, data=0)
    
    # Final Factor Calculation
    weights = {
        'volume_adjusted_momentum': 0.4,
        'std_abs_price_changes': 0.1,
        'ema_abs_price_changes': 0.1,
        'iqr_abs_price_changes': 0.1,
        'sector_relative_strength': 0.2,
        'macro_factors': 0.1
    }
    
    assert sum(weights.values()) == 1, "Weights must sum to 1"
    
    factor = (
        weights['volume_adjusted_momentum'] * volume_adjusted_momentum +
        weights['std_abs_price_changes'] * std_abs_price_changes +
        weights['ema_abs_price_changes'] * ema_abs_price_changes +
        weights['iqr_abs_price_changes'] * iqr_abs_price_changes +
        weights['sector_relative_strength'] * sector_relative_strength +
        weights['macro_factors'] * macro_factors
    )
    
    return factor.dropna()

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# macro_data = pd.read_csv('macro_data.csv', index_col='date', parse_dates=True)
# factor = heuristics_v2(df, macro_data=macro_data)
