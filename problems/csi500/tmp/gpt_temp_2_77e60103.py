import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # High-Low Range Breakout with Volume Confirmation
    # Calculate Normalized High-Low Range
    daily_range = data['high'] - data['low']
    normalized_range = daily_range / data['close'].shift(1)
    
    # Identify Breakout Days
    range_ma_5 = normalized_range.rolling(window=5, min_periods=3).mean()
    breakout_flag = normalized_range > (1.5 * range_ma_5)
    
    # Calculate Volume-Adjusted Price Change
    returns = data['close'].pct_change()
    volume_adjusted_returns = returns * np.sqrt(data['volume'])
    
    # Combine Signals - Filter price changes on breakout days
    breakout_returns = volume_adjusted_returns.where(breakout_flag, 0)
    
    # Volume-Weighted Price Reversal
    # Compute Weighted Average Price
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    weighted_price = typical_price * data['volume']
    
    # Calculate Price Momentum
    mom_5 = weighted_price.pct_change(periods=5)
    mom_10 = weighted_price.pct_change(periods=10)
    
    # Identify Reversal Patterns
    reversal_flag = (mom_5 * mom_10) < 0  # Signs differ
    
    # Volume Confirmation
    volume_ma_5 = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_ratio = data['volume'] / volume_ma_5
    reversal_signal = reversal_flag.astype(float) * volume_ratio
    
    # Intraday Pressure Accumulation
    # Calculate Buying Pressure
    buying_pressure = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Calculate Selling Pressure  
    selling_pressure = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Accumulate Pressure Difference
    pressure_diff = buying_pressure - selling_pressure
    pressure_accumulation = pressure_diff.rolling(window=5, min_periods=3).sum()
    total_volume_5d = data['volume'].rolling(window=5, min_periods=3).sum()
    normalized_pressure = pressure_accumulation / (total_volume_5d + 1e-8)
    
    # Volatility-Regime Adjusted Return
    # Calculate Realized Volatility
    returns_20d = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    
    # Identify Volatility Regimes
    vol_median_60d = returns_20d.rolling(window=60, min_periods=30).median()
    high_vol_regime = returns_20d > vol_median_60d
    
    # Calculate Regime-Specific Returns
    forward_5d_return = data['close'].pct_change(periods=5).shift(-5)
    regime_adjusted_return = forward_5d_return / (returns_20d + 1e-8)
    regime_adjusted_return = regime_adjusted_return.where(high_vol_regime, regime_adjusted_return * 0.5)
    
    # Amount-Based Order Flow Imbalance
    # Calculate Dollar Volume
    dollar_volume = data['amount']
    
    # Compute Order Flow - simplified approach using price changes
    price_change = data['close'].diff()
    up_ticks = price_change > 0
    down_ticks = price_change < 0
    
    # Calculate Imbalance Ratio
    buy_volume = dollar_volume.where(up_ticks, 0)
    sell_volume = dollar_volume.where(down_ticks, 0)
    
    buy_volume_3d = buy_volume.rolling(window=3, min_periods=2).sum()
    sell_volume_3d = sell_volume.rolling(window=3, min_periods=2).sum()
    total_volume_3d = dollar_volume.rolling(window=3, min_periods=2).sum()
    
    imbalance_ratio = (buy_volume_3d - sell_volume_3d) / (total_volume_3d + 1e-8)
    
    # Price-Volume Divergence Detection
    # Calculate Price Trend
    def linear_regression_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    price_slope_5 = linear_regression_slope(data['close'], 5)
    price_slope_10 = linear_regression_slope(data['close'], 10)
    
    # Calculate Volume Trend
    volume_slope_5 = linear_regression_slope(data['volume'], 5)
    volume_slope_10 = linear_regression_slope(data['volume'], 10)
    
    # Detect Divergence
    price_volume_divergence = (price_slope_5 * volume_slope_5 < 0).astype(float) + \
                             (price_slope_10 * volume_slope_10 < 0).astype(float)
    
    # Range Expansion Momentum
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Detect Range Expansion
    tr_ma_10 = true_range.rolling(window=10, min_periods=5).mean()
    range_expansion = true_range > tr_ma_10
    
    # Calculate Momentum Quality
    up_days = (data['close'] > data['close'].shift(1)).rolling(window=5, min_periods=3).sum()
    down_days = (data['close'] < data['close'].shift(1)).rolling(window=5, min_periods=3).sum()
    direction_consistency = up_days / (up_days + down_days + 1e-8)
    
    # Composite Signal
    range_momentum = range_expansion.astype(float) * direction_consistency
    
    # Combine all factors with equal weights
    factor = (
        breakout_returns.fillna(0) +
        reversal_signal.fillna(0) +
        normalized_pressure.fillna(0) +
        regime_adjusted_return.fillna(0) +
        imbalance_ratio.fillna(0) +
        price_volume_divergence.fillna(0) +
        range_momentum.fillna(0)
    )
    
    return factor
