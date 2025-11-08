import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility & Efficiency Framework
    # True Range and ATR
    data['TR'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['close'].shift(1)), 
                                     abs(data['low'] - data['close'].shift(1))))
    data['ATR_5'] = data['TR'].rolling(window=5).mean()
    
    # Intraday volatility
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # Efficiency Momentum
    data['price_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['efficiency_5d'] = data['price_efficiency'].rolling(window=5).mean()
    data['efficiency_10d'] = data['price_efficiency'].rolling(window=10).mean()
    data['efficiency_momentum'] = (data['efficiency_5d'] / data['efficiency_10d']) - 1
    
    # Cross-Timeframe Acceleration Analysis
    # Multi-Horizon Breakout Detection
    data['high_5d_max'] = data['high'].rolling(window=5, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['fractal_breakout_5d'] = data['high'] / data['high_5d_max'] - 1
    
    data['high_10d_max'] = data['high'].rolling(window=10, min_periods=1).apply(lambda x: x[:-1].max() if len(x) > 1 else np.nan)
    data['fractal_breakout_10d'] = data['high'] / data['high_10d_max'] - 1
    
    # Acceleration Patterns
    data['breakout_divergence'] = data['fractal_breakout_5d'] - data['fractal_breakout_10d']
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Resilience-Adaptive Signal Construction
    # Efficiency-Weighted Acceleration
    data['price_recovery_resilience'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['price_recovery_resilience'] = data.apply(
        lambda row: row['price_recovery_resilience'] if row['close'] > row['open'] else 0, axis=1
    )
    
    data['efficiency_weighted_acceleration'] = (
        data['breakout_divergence'] * data['efficiency_momentum'] * data['price_recovery_resilience']
    )
    
    # Volume-Confirmed Breakout
    data['volume_breakout_alignment'] = np.sign(data['volume_acceleration']) * np.sign(data['breakout_divergence'])
    data['volume_weighted_breakout'] = data['breakout_divergence'] * (1 + data['volume_acceleration'])
    
    # Dynamic Confirmation & Filtering
    # Spread Efficiency Divergence
    data['effective_spread'] = 2 * abs(data['close'] - (data['high'] + data['low'])/2) / ((data['high'] + data['low'])/2)
    data['spread_5d'] = data['effective_spread'].rolling(window=5).mean()
    data['spread_10d'] = data['effective_spread'].rolling(window=10).mean()
    data['spread_divergence'] = data['efficiency_momentum'] * (data['spread_5d'] / data['spread_10d'] - 1)
    
    # Amount Acceleration
    data['amount_momentum'] = data['amount'] / data['amount'].shift(3) - 1
    data['volume_amount_alignment'] = np.sign(data['amount_momentum']) * np.sign(data['volume_acceleration'])
    
    # Regime-Adaptive Synthesis
    # Volatility-Weighted Signals
    data['ATR_ratio'] = data['ATR_5'] / data['ATR_5'].rolling(window=20).mean()
    
    def get_regime_weighted_signal(row):
        if row['ATR_ratio'] > 1.2:  # High volatility
            return 0.7 * row['efficiency_weighted_acceleration'] + 0.3 * row['volume_weighted_breakout']
        elif row['ATR_ratio'] < 0.8:  # Low volatility
            return 0.3 * row['efficiency_weighted_acceleration'] + 0.3 * row['volume_weighted_breakout']
        else:  # Normal volatility
            return 0.5 * row['efficiency_weighted_acceleration'] + 0.5 * row['volume_weighted_breakout']
    
    data['base_signal'] = data.apply(get_regime_weighted_signal, axis=1)
    
    # Dynamic Persistence Enhancement
    data['signal_direction'] = np.sign(data['base_signal'])
    data['signal_consistency'] = data['signal_direction'].rolling(window=3).apply(
        lambda x: sum(x == x.iloc[-1]) if len(x) == 3 else 0
    )
    data['persistence_weighted_signal'] = data['base_signal'] * (1 + 0.1 * data['signal_consistency'])
    
    # Final Alpha Generation
    # Cross-Dimensional Confirmation
    data['volume_amount_fractal_confidence'] = data['volume_amount_alignment'] * data['amount_momentum']
    data['multi_timeframe_efficiency'] = data['efficiency_momentum'] * data['breakout_divergence']
    
    # Stability-Enhanced Output
    data['price_stability'] = 1 - abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Final factor
    data['final_factor'] = (
        data['persistence_weighted_signal'] * 
        data['price_stability'] * 
        data['volume_amount_fractal_confidence']
    )
    
    return data['final_factor']
