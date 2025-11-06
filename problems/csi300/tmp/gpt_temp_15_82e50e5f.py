import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Classification
    # Short-term volatility
    data['short_term_vol'] = (data['high'] - data['low']) / data['close']
    
    # Medium-term volatility (5-day rolling)
    data['high_roll'] = data['high'].rolling(window=5, min_periods=5).mean()
    data['low_roll'] = data['low'].rolling(window=5, min_periods=5).mean()
    data['close_roll'] = data['close'].rolling(window=5, min_periods=5).mean()
    data['medium_term_vol'] = (data['high_roll'] - data['low_roll']) / data['close_roll']
    
    # Regime state
    data['vol_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    data['regime_state'] = 0
    data.loc[data['vol_ratio'] > 1.5, 'regime_state'] = 1  # High volatility regime
    data.loc[data['vol_ratio'] < 0.5, 'regime_state'] = -1  # Low volatility regime
    
    # Volume Dynamics
    # Volume momentum
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3) - 1
    
    # Volume-volatility leadership
    vol_diff = (data['high'] - data['low'])
    data['volume_vol_leadership'] = (
        (data['volume'].shift(1) > data['volume'].shift(2)) & 
        (vol_diff > vol_diff.shift(1))
    ).astype(int) - (
        (vol_diff.shift(1) > vol_diff.shift(2)) & 
        (data['volume'] > data['volume'].shift(1))
    ).astype(int)
    
    # Trade size sensitivity
    data['amount_avg_3d'] = data['amount'].rolling(window=3, min_periods=3).mean()
    data['large_trades'] = data['amount'] > (2 * data['amount_avg_3d'])
    data['small_trades'] = data['amount'] < (0.5 * data['amount_avg_3d'])
    
    # Calculate sums using rolling windows
    data['large_trades_sum'] = data['amount'].rolling(window=3, min_periods=3).apply(
        lambda x: x[data['large_trades'].iloc[-3:].values].sum() if len(x) == 3 else np.nan
    )
    data['small_trades_sum'] = data['amount'].rolling(window=3, min_periods=3).apply(
        lambda x: x[data['small_trades'].iloc[-3:].values].sum() if len(x) == 3 else np.nan
    )
    data['trade_size_sensitivity'] = data['large_trades_sum'] / data['small_trades_sum']
    
    # Price Efficiency & Pressure
    # Regime efficiency
    data['price_range'] = data['high'] - data['low']
    data['daily_return'] = data['close'] - data['open']
    data['regime_efficiency'] = (data['daily_return'] / data['price_range']) * data['regime_state']
    
    # Pressure ratio
    data['pressure_ratio'] = (data['daily_return'] * data['volume']) / (data['price_range'] * data['volume'])
    data['pressure_ratio'] = data['pressure_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Pressure momentum
    data['pressure_momentum'] = data['pressure_ratio'] - data['pressure_ratio'].shift(2)
    
    # Signal Integration
    # Adaptive momentum
    data['regime_stability'] = data['regime_state'].rolling(window=3, min_periods=3).std().fillna(1)
    data['adaptive_momentum'] = ((data['close'] - data['close'].shift(3)) / data['close'].shift(1)) * (1 / data['regime_stability'])
    
    # Volume confirmation
    data['volume_confirmation'] = data['volume_vol_leadership'] * data['volume_momentum']
    
    # Pressure enhancement
    data['pressure_enhancement'] = data['pressure_momentum'] * data['regime_efficiency']
    
    # Final alpha
    data['alpha'] = (
        data['adaptive_momentum'] * 
        (1 + data['volume_confirmation']) * 
        (1 + data['pressure_enhancement'])
    )
    
    return data['alpha']
