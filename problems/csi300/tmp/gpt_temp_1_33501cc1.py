import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Acceleration Framework
    # Price acceleration structure
    df['ultra_short_acceleration'] = (df['close'] / df['close'].shift(1) - df['close'] / df['close'].shift(3)) / 2
    df['short_acceleration'] = (df['close'] / df['close'].shift(3) - df['close'] / df['close'].shift(8)) / 5
    df['medium_acceleration'] = (df['close'] / df['close'].shift(8) - df['close'] / df['close'].shift(15)) / 7
    
    # Volume acceleration dynamics
    df['volume_momentum'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(1) - df['volume'] / df['volume'].shift(5)) / 4
    df['volume_intensity'] = df['volume'] / (df['high'] - df['low'])
    
    # Price-Pressure Divergence Detection
    # Intraday pressure indicators
    df['opening_pressure'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['closing_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['momentum_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Divergence patterns
    df['acceleration_divergence'] = df['ultra_short_acceleration'] - df['medium_acceleration']
    df['volume_price_divergence'] = df['volume_momentum'] * df['momentum_efficiency']
    df['pressure_reversal'] = df['opening_pressure'] * df['closing_strength']
    
    # Regime-Adaptive Volatility Classification
    # Volatility characteristics
    df['daily_volatility'] = (df['high'] - df['low']) / df['close']
    df['volatility_persistence'] = df['close'].rolling(window=3).std() / df['close'].rolling(window=8).std()
    df['range_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Regime determination
    df['high_vol_regime'] = (df['volatility_persistence'] > 1.2) & (df['daily_volatility'] > 0.02)
    df['medium_vol_regime'] = (df['volatility_persistence'] >= 0.8) & (df['volatility_persistence'] <= 1.2)
    df['low_vol_regime'] = (df['volatility_persistence'] < 0.8) & (df['daily_volatility'] < 0.015)
    
    # Pattern & Reversal Synthesis
    # Reversal detection
    df['failed_breakout'] = (df['high'] > df['high'].shift(1)) & (df['close'] < df['open'])
    df['support_bounce'] = (df['low'] < df['low'].shift(1)) & (df['close'] > (df['high'] + df['low'])/2)
    
    # High-low reversal count
    high_low_reversal = []
    for i in range(len(df)):
        if i < 5:
            high_low_reversal.append(0)
        else:
            count = 0
            for j in range(1, 6):
                if (df['high'].iloc[i-j] > df['high'].iloc[i-j-1]) and (df['close'].iloc[i-j] < df['close'].iloc[i-j-1]):
                    count += 1
            high_low_reversal.append(count)
    df['high_low_reversal_count'] = high_low_reversal
    
    # Continuation confirmation
    df['momentum_alignment'] = (df['ultra_short_acceleration'] > 0) & (df['short_acceleration'] > 0)
    df['volume_confirmation'] = (df['volume_momentum'] > 0) & (df['volume_acceleration'] > 0)
    df['pressure_consistency'] = df['opening_pressure'] * df['momentum_efficiency'] > 0
    
    # Adaptive Alpha Construction
    # High volatility regime factor
    df['core_divergence'] = df['acceleration_divergence'] * df['volume_intensity']
    df['pattern_adjustment'] = df['failed_breakout'] * df['high_low_reversal_count']
    high_vol_factor = df['core_divergence'] * df['pattern_adjustment'] * df['daily_volatility']
    
    # Medium volatility regime factor
    df['acceleration_factor'] = df['ultra_short_acceleration'] * df['short_acceleration']
    df['pressure_factor'] = df['volume_price_divergence'] * df['momentum_efficiency']
    medium_vol_factor = df['acceleration_factor'] * df['pressure_factor'] * df['range_efficiency']
    
    # Low volatility regime factor
    df['convergence_factor'] = df['momentum_alignment'] * df['volume_confirmation']
    df['reversal_factor'] = df['support_bounce'] * df['pressure_reversal']
    low_vol_factor = df['convergence_factor'] * df['reversal_factor'] * df['pressure_consistency']
    
    # Final adaptive alpha factor
    alpha_factor = pd.Series(index=df.index, dtype=float)
    alpha_factor[df['high_vol_regime']] = high_vol_factor[df['high_vol_regime']]
    alpha_factor[df['medium_vol_regime']] = medium_vol_factor[df['medium_vol_regime']]
    alpha_factor[df['low_vol_regime']] = low_vol_factor[df['low_vol_regime']]
    
    # Fill any remaining NaN values with the medium volatility factor
    alpha_factor = alpha_factor.fillna(medium_vol_factor)
    
    return alpha_factor
