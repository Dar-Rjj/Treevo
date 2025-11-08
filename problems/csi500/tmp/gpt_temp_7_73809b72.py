import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Regime-Adaptive Momentum with Volume Efficiency alpha factor
    """
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components
    data['gap_magnitude'] = data['open'] / data['close'].shift(1) - 1
    data['gap_direction'] = np.sign(data['gap_magnitude'])
    data['gap_acceleration'] = data['gap_magnitude'] - data['gap_magnitude'].shift(1)
    data['gap_fill_asymmetry'] = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'])
    
    # Intraday components
    data['intraday_direction'] = np.sign(data['close'] - data['open'])
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    data['morning_afternoon_momentum'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] + data['low'])/2 - data['open'])
    data['asymmetric_volatility'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Multi-period components
    data['short_term_momentum'] = data['close'] / data['close'].shift(2) - 1
    data['medium_term_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_alignment'] = np.sign(data['short_term_momentum']) * np.sign(data['medium_term_momentum'])
    data['momentum_strength'] = np.abs(data['short_term_momentum']) + np.abs(data['medium_term_momentum'])
    
    # Volume efficiency components
    data['direction_efficiency'] = np.sign(data['close'] - data['open']) * np.sign(data['volume'] - data['volume'].shift(1))
    data['magnitude_efficiency'] = np.abs(data['close'] - data['open']) * np.abs(data['volume'] - data['volume'].shift(1))
    data['price_impact_volume'] = np.abs(data['close'] - data['open']) / data['volume']
    data['volume_efficiency_ratio'] = np.abs(data['close'] - data['open']) / (data['volume'] / data['volume'].rolling(window=5).mean())
    
    # Calculate rolling statistics
    for window in [5]:
        data[f'vol_avg_{window}'] = (data['high'] - data['low']).rolling(window=window).mean()
        data[f'volume_avg_{window}'] = data['volume'].rolling(window=window).mean()
        data[f'volume_std_{window}'] = data['volume'].rolling(window=window).std()
    
    # Persistence calculations
    for col in ['gap_direction', 'direction_efficiency', 'volume']:
        data[f'{col}_persistence'] = 0
        current_streak = 0
        for i in range(1, len(data)):
            if data[col].iloc[i] == data[col].iloc[i-1]:
                current_streak += 1
            else:
                current_streak = 0
            data[f'{col}_persistence'].iloc[i] = current_streak
    
    data['efficiency_persistence'] = 0
    current_streak = 0
    for i in range(1, len(data)):
        if data['range_efficiency'].iloc[i] > 0.5:
            current_streak += 1
        else:
            current_streak = 0
        data['efficiency_persistence'].iloc[i] = current_streak
    
    # Regime detection
    data['volatility_trend'] = (data['high'] - data['low']) / data['vol_avg_5']
    data['volume_level'] = data['volume'] / data['volume_avg_5']
    data['volume_trend'] = data['volume'] / data['volume'].shift(2) - 1
    data['volatility_clustering'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Regime persistence
    data['momentum_regime_persistence'] = 0
    current_streak = 0
    for i in range(1, len(data)):
        if data['momentum_alignment'].iloc[i] == data['momentum_alignment'].iloc[i-1]:
            current_streak += 1
        else:
            current_streak = 0
        data['momentum_regime_persistence'].iloc[i] = current_streak
    
    data['volume_persistence'] = 0
    current_streak = 0
    for i in range(1, len(data)):
        if data['volume_level'].iloc[i] > 1:
            current_streak += 1
        else:
            current_streak = 0
        data['volume_persistence'].iloc[i] = current_streak
    
    # Calculate factor for each day
    for i in range(5, len(data)):
        if pd.isna(data.iloc[i][['gap_magnitude', 'range_efficiency', 'volume']]).any():
            continue
            
        # Core Asymmetric Signal
        gap_momentum = data['gap_magnitude'].iloc[i] * (1 + data['gap_direction_persistence'].iloc[i] / 10)
        intraday_efficiency = gap_momentum * data['range_efficiency'].iloc[i]
        asymmetric_enhancement = intraday_efficiency * (1 + data['morning_afternoon_momentum'].iloc[i])
        persistence_boost = asymmetric_enhancement * (1 + data['direction_efficiency_persistence'].iloc[i] / 15)
        
        # Volume Confirmation Layer
        direction_confirmation = persistence_boost * (1 + data['direction_efficiency'].iloc[i] * 0.5)
        efficiency_confirmation = direction_confirmation * (1 + data['volume_efficiency_ratio'].iloc[i])
        persistence_confirmation = efficiency_confirmation * (1 + data['efficiency_persistence'].iloc[i] / 10)
        volume_quality = persistence_confirmation * (1 + data['volume_persistence'].iloc[i] / 8)
        
        # Multi-Regime Adaptive Scaling
        core_factor = volume_quality
        
        # Momentum Regime Adaptation
        if data['momentum_alignment'].iloc[i] == 1:
            if data['momentum_strength'].iloc[i] > 0.05:
                core_factor *= 1.8  # Strong Bullish
            elif data['momentum_strength'].iloc[i] < -0.05:
                core_factor *= 1.8  # Strong Bearish
        elif data['momentum_alignment'].iloc[i] == -1:
            core_factor *= 0.4  # Mixed Momentum
        
        if data['momentum_regime_persistence'].iloc[i] >= 3:
            core_factor += 0.02 * data['momentum_regime_persistence'].iloc[i]
        elif data['momentum_regime_persistence'].iloc[i] <= 1:
            core_factor *= 0.7
        
        # Volatility Regime Adaptation
        vol_trend = data['volatility_trend'].iloc[i]
        if vol_trend > 1.2:
            core_factor *= 0.6  # High Volatility
        elif vol_trend < 0.8:
            core_factor *= 1.4  # Low Volatility
        else:
            core_factor *= 1.0  # Normal Volatility
            
        if data['volatility_clustering'].iloc[i] > 1.2:
            core_factor *= 0.8  # Increasing Volatility
        elif data['volatility_clustering'].iloc[i] < 0.8:
            core_factor *= 1.2  # Decreasing Volatility
        
        # Volume Regime Adaptation
        vol_level = data['volume_level'].iloc[i]
        if vol_level > 1.2 and data['volume_persistence'].iloc[i] >= 2:
            core_factor *= 1.6  # High Volume with Persistence
        elif vol_level < 0.8:
            core_factor *= 0.5  # Low Volume
        else:
            core_factor *= 1.0  # Normal Volume
            
        if abs(data['volume_trend'].iloc[i]) > 0.3:
            core_factor *= 1.3  # Strong Volume Trend
        elif abs(data['volume_trend'].iloc[i]) < 0.1:
            core_factor *= 0.7  # Weak Volume Trend
        
        # Signal Validation & Quality Control
        min_volume_ok = data['volume'].iloc[i] > 0.5 * data['volume_avg_5'].iloc[i]
        efficiency_ok = data['range_efficiency'].iloc[i] > 0.1
        persistence_ok = data['direction_efficiency_persistence'].iloc[i] >= 1
        gap_ok = abs(data['gap_magnitude'].iloc[i]) < 0.1
        
        if min_volume_ok and efficiency_ok and persistence_ok and gap_ok:
            factor.iloc[i] = core_factor
        else:
            factor.iloc[i] = 0
    
    return factor
