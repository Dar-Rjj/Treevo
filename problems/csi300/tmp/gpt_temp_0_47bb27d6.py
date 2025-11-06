import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Momentum Efficiency with Gap Reversion Signals
    """
    data = df.copy()
    
    # Calculate Overnight Price Gap
    data['gap_ratio'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Identify Extreme Gap Conditions (top/bottom 10%)
    data['gap_percentile'] = data['gap_ratio'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else 0.5
    )
    data['extreme_gap'] = ((data['gap_percentile'] >= 0.9) | (data['gap_percentile'] <= 0.1)).astype(int)
    
    # Calculate Momentum Acceleration
    data['return_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['return_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['momentum_acceleration'] = np.where(
        data['return_20d'] != 0,
        (data['return_5d'] / data['return_20d']) - 1,
        0
    )
    
    # Calculate Intraday Price Efficiency
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_efficiency'] = np.where(
        data['intraday_range'] != 0,
        (data['close'] - data['open']) / data['intraday_range'],
        0
    )
    
    # Combine Gap Reversion with Momentum Efficiency
    data['gap_efficiency'] = data['gap_ratio'] * data['intraday_efficiency']
    data['combined_signal'] = data['gap_efficiency'] * data['momentum_acceleration']
    
    # Apply extreme gap filter to enhance signal strength
    data['enhanced_signal'] = data['combined_signal'] * (1 + data['extreme_gap'] * 0.5)
    
    # Calculate Volatility Filter (Average True Range)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    
    # Calculate volatility Z-score
    data['volatility_mean'] = data['atr_10d'].rolling(window=20, min_periods=10).mean()
    data['volatility_std'] = data['atr_10d'].rolling(window=20, min_periods=10).std()
    data['volatility_zscore'] = np.where(
        data['volatility_std'] != 0,
        (data['atr_10d'] - data['volatility_mean']) / data['volatility_std'],
        0
    )
    
    # Calculate Volume Confirmation
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_spike'] = np.where(
        data['volume_5d_avg'] != 0,
        data['volume'] / data['volume_5d_avg'],
        1
    )
    data['volume_strength'] = np.log(data['volume'] + 1)
    
    # Final Signal Generation
    data['final_signal'] = (
        data['enhanced_signal'] * 
        data['volatility_zscore'] * 
        data['volume_spike'] * 
        data['volume_strength']
    )
    
    # Apply extreme gap condition as final amplifier
    data['alpha_factor'] = data['final_signal'] * (1 + data['extreme_gap'] * 0.3)
    
    return data['alpha_factor']
