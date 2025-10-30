import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Efficiency Component
    df['price_efficiency_short'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001)
    df['price_efficiency_medium'] = (df['close'].shift(3) - df['open'].shift(3)) / (df['high'].shift(3) - df['low'].shift(3) + 0.0001)
    df['efficiency_divergence'] = df['price_efficiency_short'] - df['price_efficiency_medium']
    
    # Volume-Price Alignment Component
    df['volume_price_movement'] = df['volume'] * (df['close'] - df['open']) / df['close']
    df['volume_price_alignment_3d'] = df['volume_price_movement'].rolling(window=3, min_periods=1).mean()
    df['alignment_momentum'] = df['volume_price_movement'] / (df['volume_price_alignment_3d'] + 0.0001)
    
    # Efficiency Convergence Detection
    df['direction_consistency'] = np.where(
        np.sign(df['efficiency_divergence']) == np.sign(df['alignment_momentum']), 1.0,
        np.where(
            (df['efficiency_divergence'] == 0) | (df['alignment_momentum'] == 0), 0.5, 0.0
        )
    )
    df['convergence_strength'] = df['efficiency_divergence'] * df['alignment_momentum'] * df['direction_consistency']
    
    # Dynamic Regime and Volatility Analysis
    df['amplitude_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['open']),
            np.abs(df['low'] - df['open'])
        )
    )
    df['regime_shift'] = df['amplitude_range'] > df['amplitude_range'].rolling(window=3, min_periods=1).median()
    df['regime_sensitivity'] = np.where(df['regime_shift'], 2.0, 1.2)
    
    df['daily_range'] = df['high'] - df['low']
    df['avg_daily_range_5d'] = df['daily_range'].rolling(window=5, min_periods=1).mean()
    df['volatility_adjustment'] = 1 / (df['avg_daily_range_5d'] + 0.0001)
    
    df['dynamic_scaling'] = df['regime_sensitivity'] * df['volatility_adjustment']
    df['scaled_convergence'] = df['convergence_strength'] * df['dynamic_scaling']
    
    # Trade Size and Amount Coordination
    df['daily_trade_size'] = df['amount'] / (df['volume'] + 0.0001)
    df['avg_trade_size_5d'] = df['daily_trade_size'].rolling(window=5, min_periods=1).mean()
    df['trade_size_ratio'] = df['daily_trade_size'] / (df['avg_trade_size_5d'] + 0.0001)
    
    df['amount_concentration'] = df['amount'] / (df['amount'] + df['amount'].shift(1) + df['amount'].shift(2) + 0.0001)
    df['amount_momentum'] = (df['amount'] / (df['amount'].shift(1) + 0.0001)) * (df['amount'].shift(1) / (df['amount'].shift(2) + 0.0001))
    
    amount_range = df['amount'].rolling(window=3, min_periods=1).apply(lambda x: x.max() - x.min())
    amount_mean = df['amount'].rolling(window=3, min_periods=1).mean()
    df['amount_efficiency'] = df['amount_concentration'] / ((amount_range / (amount_mean + 0.0001)) + 0.0001)
    
    df['trade_size_direction'] = np.sign(df['trade_size_ratio'] - 1)
    df['trade_size_weight'] = df['trade_size_ratio'] * df['trade_size_direction']
    df['trade_size_alignment'] = df['trade_size_weight'] * df['amount_efficiency']
    
    # Opening Strength and Trend Enhancement
    df['min_low_open'] = np.minimum(df['low'], df['open'])
    df['max_high_open'] = np.maximum(df['high'], df['open'])
    df['opening_strength'] = (df['close'] - df['min_low_open']) / (df['max_high_open'] - df['min_low_open'] + 0.0001)
    df['opening_adjustment'] = 1 + 0.3 * df['opening_strength']
    
    df['price_momentum_3d'] = (df['close'] - df['close'].shift(3)) / (df['close'].shift(3) + 0.0001)
    df['trend_magnitude'] = np.abs(df['price_momentum_3d'])
    
    df['trend_weighted_opening'] = df['trend_magnitude'] * df['opening_adjustment']
    df['trend_confirmed_convergence'] = df['scaled_convergence'] * df['trend_weighted_opening']
    
    # Integrated Alpha Factor Generation
    df['core_convergence_signal'] = df['trend_confirmed_convergence'] * df['trade_size_alignment']
    
    # Filter extreme values using MAD
    recent_mad = df['core_convergence_signal'].rolling(window=15, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    df['alpha_factor'] = np.clip(
        df['core_convergence_signal'], 
        -2.5 * recent_mad, 
        2.5 * recent_mad
    )
    
    return df['alpha_factor']
