import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['close_change'] = df['close'].diff()
    
    # Dual-Timeframe Efficiency Calculation
    # Short-Term Efficiency (5-day)
    df['net_movement_5d'] = df['close'] - df['close'].shift(5)
    df['total_movement_5d'] = df['close_change'].abs().rolling(window=5, min_periods=5).sum()
    df['efficiency_5d'] = df['net_movement_5d'] / df['total_movement_5d']
    
    # Medium-Term Efficiency (15-day)
    df['net_movement_15d'] = df['close'] - df['close'].shift(15)
    df['total_movement_15d'] = df['close_change'].abs().rolling(window=15, min_periods=15).sum()
    df['efficiency_15d'] = df['net_movement_15d'] / df['total_movement_15d']
    
    # Efficiency Momentum Analysis
    df['efficiency_acceleration'] = (df['efficiency_5d'] - df['efficiency_15d']) * np.sign(df['close'] - df['close'].shift(5))
    
    # Efficiency Persistence
    positive_accel = (df['efficiency_acceleration'] > 0).astype(int)
    negative_accel = (df['efficiency_acceleration'] < 0).astype(int)
    
    # Calculate consecutive positive and negative days
    pos_persistence = positive_accel * (positive_accel.groupby((positive_accel != positive_accel.shift()).cumsum()).cumcount() + 1)
    neg_persistence = negative_accel * (negative_accel.groupby((negative_accel != negative_accel.shift()).cumsum()).cumcount() + 1)
    
    df['efficiency_persistence'] = pos_persistence - neg_persistence
    
    # Volatility Context Integration
    df['high_low_range'] = df['high'] - df['low']
    range_volatility = df['high_low_range'].rolling(window=20, min_periods=20).std()
    return_volatility = df['returns'].rolling(window=20, min_periods=20).std()
    df['combined_volatility'] = np.sqrt(range_volatility * return_volatility)
    
    # Volatility-Weighted Signal with exponential weighting
    volatility_weighted = df['efficiency_acceleration'] / df['combined_volatility']
    volatility_weighted = volatility_weighted * df['efficiency_persistence']
    
    # Apply exponential weighting (recent days higher)
    weights = np.exp(np.linspace(-1, 0, len(volatility_weighted)))
    weights = weights / weights.sum()
    df['volatility_weighted_signal'] = volatility_weighted.rolling(window=len(weights), min_periods=1).apply(
        lambda x: np.sum(x * weights[:len(x)]), raw=False
    )
    
    # Volume-Price Confirmation
    df['daily_vwap'] = (df['high'] + df['low'] + df['close']) / 3 * df['volume']
    df['price_deviation'] = (df['close'] - df['daily_vwap']) / abs(df['close'])
    
    # Volume Trend Alignment
    volume_ma_10 = df['volume'].rolling(window=10, min_periods=10).mean()
    volume_ma_20 = df['volume'].rolling(window=20, min_periods=20).mean()
    df['volume_trend'] = (volume_ma_10 / volume_ma_20 - 1)
    
    # Final Alpha Construction
    alpha = df['volatility_weighted_signal'] * df['price_deviation'] * df['volume_trend']
    
    # Apply mean reversion filter for extreme values
    alpha_rolling_mean = alpha.rolling(window=20, min_periods=20).mean()
    alpha_rolling_std = alpha.rolling(window=20, min_periods=20).std()
    alpha_filtered = (alpha - alpha_rolling_mean) / alpha_rolling_std
    alpha_filtered = np.tanh(alpha_filtered)  # Soft bound the values
    
    return alpha_filtered
