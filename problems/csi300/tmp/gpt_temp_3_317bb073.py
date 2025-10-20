import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration Analysis
    # Compute multi-timeframe momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Derive acceleration as momentum difference and rate of change
    data['momentum_diff'] = data['momentum_5d'] - data['momentum_10d']
    data['acceleration'] = data['momentum_diff'] - data['momentum_diff'].shift(3)
    data['acceleration_roc'] = data['acceleration'] / (abs(data['momentum_diff'].shift(3)) + 1e-8)
    
    # Volume Asymmetry Patterns
    # Calculate up-day/down-day volume sensitivity
    data['price_change'] = data['close'] / data['close'].shift(1) - 1
    data['is_up_day'] = (data['price_change'] > 0).astype(int)
    data['is_down_day'] = (data['price_change'] < 0).astype(int)
    
    # Calculate rolling volume ratios for up and down days
    data['up_day_volume'] = data['volume'] * data['is_up_day']
    data['down_day_volume'] = data['volume'] * data['is_down_day']
    
    up_volume_ma = data['up_day_volume'].rolling(window=10, min_periods=5).mean()
    down_volume_ma = data['down_day_volume'].rolling(window=10, min_periods=5).mean()
    data['volume_sensitivity_ratio'] = up_volume_ma / (down_volume_ma + 1e-8)
    
    # Detect divergence between volume trends and price acceleration
    volume_trend = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=20).mean() - 1
    data['volume_accel_divergence'] = np.sign(data['acceleration']) * np.sign(volume_trend)
    data['divergence_strength'] = abs(data['acceleration']) * abs(volume_trend) * data['volume_accel_divergence']
    
    # Breakout Confirmation
    # Assess intraday range dynamics
    data['intraday_range'] = (data['high'] - data['low']) / data['close']
    data['range_ratio'] = data['intraday_range'] / data['intraday_range'].rolling(window=10).mean()
    
    # Evaluate volume breakout conditions
    volume_ma_short = data['volume'].rolling(window=5).mean()
    volume_ma_long = data['volume'].rolling(window=20).mean()
    data['volume_breakout'] = volume_ma_short / volume_ma_long - 1
    
    # Volume persistence indicator
    data['volume_persistence'] = (data['volume'] > data['volume'].rolling(window=5).mean()).rolling(window=3).sum()
    
    # Composite Alpha Generation
    # Combine momentum acceleration with volume asymmetry weighting
    momentum_component = data['acceleration_roc'] * np.tanh(data['volume_sensitivity_ratio'] - 1)
    
    # Integrate divergence detection for directional bias
    divergence_component = data['divergence_strength'] * np.sign(data['acceleration'])
    
    # Apply breakout validation for signal enhancement
    breakout_component = data['range_ratio'] * data['volume_breakout'] * (data['volume_persistence'] / 3)
    
    # Final alpha factor
    alpha = (momentum_component * 0.4 + 
             divergence_component * 0.35 + 
             breakout_component * 0.25)
    
    # Normalize the alpha factor
    alpha_rank = alpha.rolling(window=20, min_periods=10).apply(
        lambda x: (x.rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan), 
        raw=False
    )
    
    return alpha_rank
