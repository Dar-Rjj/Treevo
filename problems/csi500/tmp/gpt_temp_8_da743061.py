import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence with Volume Confirmation
    # Short-term momentum (5-day)
    short_term_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Medium-term momentum (20-day)
    medium_term_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Volume trend (current volume vs 5-day MA)
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_trend = data['volume'] / volume_ma_5
    
    # Momentum divergence with volume confirmation
    momentum_divergence = (short_term_momentum * volume_trend) / (medium_term_momentum + 1e-8)
    
    # Volatility-Adjusted Mean Reversion
    # Price deviation from 10-day MA
    price_ma_10 = data['close'].rolling(window=10, min_periods=1).mean()
    price_deviation = (data['close'] - price_ma_10) / price_ma_10
    
    # Price range
    price_range = data['high'] - data['low']
    
    # True Range calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Normalize reversion by volatility
    volatility_adjusted_reversion = (price_deviation / (true_range + 1e-8)) * (price_range / data['close'])
    
    # Liquidity-Efficient Breakout Detection
    # Resistance and support levels
    resistance_level = data['high'].rolling(window=20, min_periods=1).max()
    support_level = data['low'].rolling(window=20, min_periods=1).min()
    
    # Breakout strength
    high_breakout = (data['high'] - resistance_level) / resistance_level
    low_breakout = (support_level - data['low']) / support_level
    breakout_strength = high_breakout - low_breakout
    
    # Volume efficiency (volume per price move)
    price_change = abs(data['close'] - data['close'].shift(1))
    volume_efficiency = data['volume'] / (price_change + 1e-8)
    
    # Liquidity-efficient breakout
    liquidity_breakout = breakout_strength * volume_efficiency
    
    # Order Flow Imbalance Factor
    # Close-to-open gap
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday range utilization
    intraday_range = data['high'] - data['low']
    range_utilization = (data['close'] - data['low']) / (intraday_range + 1e-8)
    
    # Volume concentration
    volume_std_5 = data['volume'].rolling(window=5, min_periods=1).std()
    volume_mean_5 = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_concentration = volume_std_5 / (volume_mean_5 + 1e-8)
    
    # Order flow imbalance
    order_flow_imbalance = (gap * range_utilization) / (volume_concentration + 1e-8)
    
    # Trend Persistence with Volatility Regime
    # Calculate trend slope using linear regression over 10 days
    def calculate_slope(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    trend_slope = data['close'].rolling(window=10, min_periods=2).apply(calculate_slope, raw=False)
    
    # Calculate R-squared for trend consistency
    def calculate_r_squared(series):
        if len(series) < 2:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        coeffs = np.polyfit(x, y, 1)
        p = np.poly1d(coeffs)
        yhat = p(x)
        ybar = np.sum(y) / len(y)
        ssreg = np.sum((yhat - ybar)**2)
        sstot = np.sum((y - ybar)**2)
        return ssreg / sstot if sstot != 0 else 0
    
    trend_consistency = data['close'].rolling(window=10, min_periods=2).apply(calculate_r_squared, raw=False)
    
    # Volatility clustering (autocorrelation of squared returns)
    returns = data['close'].pct_change()
    squared_returns = returns ** 2
    volatility_clustering = squared_returns.rolling(window=10, min_periods=2).corr(squared_returns.shift(1))
    
    # Trend persistence with volatility adjustment
    trend_persistence = (trend_slope * trend_consistency) / (abs(volatility_clustering) + 1e-8)
    
    # Combine all factors with equal weighting
    combined_factor = (
        momentum_divergence.fillna(0) +
        volatility_adjusted_reversion.fillna(0) +
        liquidity_breakout.fillna(0) +
        order_flow_imbalance.fillna(0) +
        trend_persistence.fillna(0)
    )
    
    return combined_factor
