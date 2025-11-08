import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Adaptive Momentum with Volume Regime Alignment alpha factor
    """
    data = df.copy()
    
    # Core Momentum Components
    # Price Momentum
    data['daily_return'] = data['close'] - data['close'].shift(1)
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Momentum
    data['volume_change'] = data['volume'] / data['volume'].shift(1)
    data['volume_adjusted_return'] = data['daily_return'] * data['volume']
    
    # Regime Detection
    # Volatility Regime
    data['short_term_vol'] = data['close'].rolling(window=5).std()
    data['long_term_vol'] = data['close'].rolling(window=20).std()
    data['volatility_ratio'] = data['short_term_vol'] / data['long_term_vol']
    
    # Volume Regime
    data['volume_diff_sign'] = np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_trend'] = data['volume_diff_sign'].rolling(window=5).sum()
    data['volume_level'] = data['volume'].rolling(window=10).mean()
    data['volume_regime_signal'] = data['volume_trend'] * data['volume_level']
    
    # Momentum Persistence Analysis
    # Direction Persistence
    data['momentum_direction'] = np.sign(data['daily_return'])
    data['consecutive_days'] = 0
    
    # Calculate consecutive days with same direction
    for i in range(1, len(data)):
        if data['momentum_direction'].iloc[i] == data['momentum_direction'].iloc[i-1]:
            data.loc[data.index[i], 'consecutive_days'] = data['consecutive_days'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'consecutive_days'] = 1
    
    # Volume-Momentum Alignment
    data['alignment_signal'] = data['momentum_direction'] * np.sign(data['volume'] - data['volume'].shift(1))
    data['alignment_streak'] = 0
    
    # Calculate alignment streak
    for i in range(1, len(data)):
        if data['alignment_signal'].iloc[i] > 0:
            data.loc[data.index[i], 'alignment_streak'] = data['alignment_streak'].iloc[i-1] + 1
        else:
            data.loc[data.index[i], 'alignment_streak'] = 0
    
    data['alignment_strength'] = data['alignment_streak'] * abs(data['volume'] - data['volume'].shift(1))
    
    # Adaptive Factor Construction
    # Base Momentum Factor
    data['core_momentum'] = data['daily_return'] * data['consecutive_days']
    data['volume_confirmation'] = data['core_momentum'] * data['alignment_strength']
    
    # Regime-Based Adjustment
    data['volatility_adjustment'] = np.where(
        data['volatility_ratio'] > 1,
        2 - data['volatility_ratio'],  # High volatility adjustment
        data['volatility_ratio']       # Low volatility adjustment
    )
    
    data['volume_regime_adjustment'] = np.where(
        data['volume_regime_signal'] > 0,
        1 + data['volume_regime_signal'] / 10,        # Rising volume boost
        1 - abs(data['volume_regime_signal']) / 10    # Falling volume penalty
    )
    
    # Recent Trend Confirmation
    data['three_day_momentum'] = data['close'].diff(periods=3)
    data['trend_alignment'] = np.sign(data['three_day_momentum']) * np.sign(data['daily_return'])
    data['final_multiplier'] = 1 + (data['trend_alignment'] * 0.2)
    
    # Final Alpha Output
    # Adaptive Momentum Factor
    data['base_factor'] = data['volume_confirmation'] * data['volatility_adjustment'] * data['volume_regime_adjustment']
    data['adaptive_momentum_factor'] = data['base_factor'] * data['final_multiplier']
    
    # Return the factor values
    return data['adaptive_momentum_factor']
