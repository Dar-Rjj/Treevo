import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Adjusted Volume Acceleration
    # Calculate Price Momentum
    data['momentum_5'] = data['close'].pct_change(5)
    data['momentum_10'] = data['close'].pct_change(10)
    data['momentum_divergence'] = data['momentum_5'] - data['momentum_10']
    
    # Calculate Volume Acceleration
    data['volume_change_5'] = data['volume'].pct_change(5)
    data['volume_change_10'] = data['volume'].pct_change(10)
    data['volume_acceleration'] = data['volume_change_5'] / data['volume_change_10'].replace(0, np.nan)
    
    # Combine Signals
    data['momentum_volume_signal'] = data['momentum_divergence'] * data['volume_acceleration']
    data['returns_20d_vol'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    factor1 = np.sign(data['momentum_volume_signal']) / data['returns_20d_vol'].replace(0, np.nan)
    
    # Volatility-Regime Adjusted Amihud Ratio
    # Calculate Amihud Illiquidity
    data['daily_return'] = data['close'].pct_change()
    data['amihud'] = abs(data['daily_return']) / data['volume'].replace(0, np.nan)
    
    # Assess Volatility Regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['avg_daily_range_20d'] = data['daily_range'].rolling(window=20, min_periods=10).mean()
    
    # Adjust for Regime
    data['cum_return_5d'] = data['close'].pct_change(5)
    factor2 = np.log(data['amihud'] / data['avg_daily_range_20d'].replace(0, np.nan)) * np.sign(data['cum_return_5d'])
    
    # Opening Gap Mean Reversion with Volume Confirmation
    # Calculate Opening Gap
    data['prev_close'] = data['close'].shift(1)
    data['gap'] = (data['open'] - data['prev_close']) / data['prev_close']
    
    # Assess Volume Pattern
    data['avg_volume_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ratio'] = data['volume'] / data['avg_volume_5d'].replace(0, np.nan)
    
    # Generate Signal
    factor3 = np.zeros(len(data))
    long_condition = (data['gap'] < -0.02) & (data['volume_ratio'] < 0.8)
    short_condition = (data['gap'] > 0.02) & (data['volume_ratio'] < 0.8)
    factor3[long_condition] = data['gap'][long_condition] * -1  # Long signal (positive for negative gap)
    factor3[short_condition] = data['gap'][short_condition] * -1  # Short signal (negative for positive gap)
    
    # Intraday Strength Persistence
    # Calculate Intraday Strength
    data['high_low_range'] = data['high'] - data['low']
    data['intraday_strength'] = (data['close'] - data['low']) / data['high_low_range'].replace(0, np.nan)
    
    # Assess Persistence
    data['avg_strength_3d'] = data['intraday_strength'].rolling(window=3, min_periods=2).mean()
    data['strength_trend'] = data['intraday_strength'] - data['avg_strength_3d']
    data['strength_vol_5d'] = data['intraday_strength'].rolling(window=5, min_periods=3).std()
    
    # Generate Composite
    strength_component = data['strength_trend'] * data['intraday_strength']
    strength_component = strength_component / data['strength_vol_5d'].replace(0, np.nan)
    factor4 = np.tanh(strength_component)
    
    # Volume-Weighted Price Level Efficiency
    # Calculate VWAP Efficiency
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    data['efficiency'] = abs(data['close'] - data['vwap']) / data['high_low_range'].replace(0, np.nan)
    
    # Assess Volume Distribution
    data['volume_skew_3d'] = data['volume'].rolling(window=3, min_periods=2).skew()
    data['volume_kurt_5d'] = data['volume'].rolling(window=5, min_periods=3).kurt()
    
    # Combine Metrics
    factor5 = (data['efficiency'] * data['volume_skew_3d']) + data['volume_kurt_5d']
    factor5 = 1 / (1 + abs(factor5))
    
    # Regime-Switching Momentum Breakout
    # Identify Regime
    data['volatility_20d'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    
    # Calculate trend slope using linear regression over 50 days
    def calc_trend_slope(series):
        if len(series) < 50:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series.values, 1)[0]
    
    data['trend_slope'] = data['close'].rolling(window=50, min_periods=30).apply(calc_trend_slope, raw=False)
    data['vol_quintile'] = pd.qcut(data['volatility_20d'], 5, labels=False, duplicates='drop') + 1
    data['regime'] = np.sign(data['trend_slope']) * data['vol_quintile']
    
    # Detect Breakout
    data['high_20d'] = data['high'].rolling(window=20, min_periods=10).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=10).min()
    data['breakout_signal'] = (data['close'] - data['high_20d']) / (data['high_20d'] - data['low_20d']).replace(0, np.nan)
    
    # Generate Factor
    data['avg_volume_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_confirmation'] = data['volume'] / data['avg_volume_10d'].replace(0, np.nan)
    factor6 = data['regime'] * data['breakout_signal'] * data['volume_confirmation'] * data['momentum_5']
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'f1': factor1,
        'f2': factor2,
        'f3': factor3,
        'f4': factor4,
        'f5': factor5,
        'f6': factor6
    })
    
    # Z-score normalization for each factor
    for col in factors.columns:
        factors[col] = (factors[col] - factors[col].mean()) / factors[col].std()
    
    # Equal-weighted combination
    final_factor = factors.mean(axis=1)
    
    return final_factor
