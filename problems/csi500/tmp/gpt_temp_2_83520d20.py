import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence with Volume Confirmation
    # Price Momentum Component
    short_momentum = data['close'] / data['close'].shift(5) - 1
    medium_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Volume Confirmation
    volume_ma5 = data['volume'].rolling(window=5).mean()
    volume_trend = data['volume'] / volume_ma5
    
    # Combined Momentum Factor
    momentum_factor = short_momentum * volume_trend / (medium_momentum + 1e-8)
    
    # Volatility-Adjusted Mean Reversion
    # Price Reversion Component
    close_ma10 = data['close'].rolling(window=10).mean()
    price_deviation = data['close'] - close_ma10
    price_range_ratio = data['high'] / data['low']
    
    # Volatility Adjustment
    prev_close = data['close'].shift(1)
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - prev_close)
    tr3 = abs(data['low'] - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Combined Mean Reversion Factor
    mean_reversion_factor = price_deviation * price_range_ratio / (true_range + 1e-8)
    
    # Liquidity-Efficient Breakout
    # Breakout Identification
    high_rollmax20 = data['high'].rolling(window=20).max()
    low_rollmin20 = data['low'].rolling(window=20).min()
    resistance_break = data['high'] / high_rollmax20
    support_break = low_rollmin20 / data['low']
    breakout_strength = resistance_break - support_break
    
    # Liquidity Efficiency
    price_change = abs(data['close'] - data['close'].shift(1))
    volume_efficiency = data['volume'] / (price_change + 1e-8)
    
    # Combined Breakout Factor
    breakout_factor = breakout_strength * volume_efficiency
    
    # Order Flow Imbalance
    # Intraday Price Pressure
    close_open_gap = data['close'].shift(1) / data['open'] - 1
    high_low_range = data['high'] - data['low']
    range_utilization = (data['close'] - data['low']) / (high_low_range + 1e-8)
    
    # Volume Distribution
    volume_std5 = data['volume'].rolling(window=5).std()
    volume_mean5 = data['volume'].rolling(window=5).mean()
    volume_concentration = volume_std5 / (volume_mean5 + 1e-8)
    
    # Combined Order Flow Factor
    order_flow_factor = close_open_gap * range_utilization / (volume_concentration + 1e-8)
    
    # Trend Persistence with Volatility Regime
    # Trend Quality
    def linear_regression_slope(x):
        if len(x) < 10:
            return np.nan
        y = np.arange(len(x))
        slope, _, r_value, _, _ = stats.linregress(y, x)
        return slope, r_value**2
    
    # Calculate rolling regression
    trend_results = data['close'].rolling(window=10).apply(
        lambda x: linear_regression_slope(x)[0] if len(x) == 10 else np.nan, 
        raw=False
    )
    r_squared_results = data['close'].rolling(window=10).apply(
        lambda x: linear_regression_slope(x)[1] if len(x) == 10 else np.nan, 
        raw=False
    )
    
    trend_slope = trend_results
    trend_consistency = r_squared_results
    
    # Volatility Adjustment
    returns = data['close'].pct_change()
    squared_returns = returns ** 2
    volatility_clustering = squared_returns.rolling(window=10).corr(squared_returns.shift(1))
    
    # Combined Trend Factor
    trend_factor = trend_slope * trend_consistency / (volatility_clustering + 1e-8)
    
    # Combine all factors with equal weights
    combined_factor = (
        momentum_factor.fillna(0) + 
        mean_reversion_factor.fillna(0) + 
        breakout_factor.fillna(0) + 
        order_flow_factor.fillna(0) + 
        trend_factor.fillna(0)
    )
    
    return combined_factor
