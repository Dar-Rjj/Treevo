import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence Factor
    """
    data = df.copy()
    
    # Price and Volume changes
    data['price_change'] = data['close'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['price_magnitude'] = abs(data['price_change'])
    data['volume_magnitude'] = abs(data['volume_change'])
    
    # Divergence Classification
    def classify_divergence(row):
        if pd.isna(row['price_change']) or pd.isna(row['volume_change']):
            return 0
        if row['price_change'] > 0 and row['volume_change'] < 0:
            return 1  # Positive divergence
        elif row['price_change'] < 0 and row['volume_change'] > 0:
            return -1  # Negative divergence
        else:
            return 0  # Confirmation
    
    data['divergence_signal'] = data.apply(classify_divergence, axis=1)
    data['divergence_intensity'] = data['price_magnitude'] * data['volume_magnitude'] * data['divergence_signal']
    
    # Multi-Timeframe Divergence Assessment
    data['ultra_short_div'] = data['divergence_signal']
    data['short_term_div'] = data['divergence_signal'].rolling(window=3, min_periods=1).sum()
    data['medium_term_div'] = data['divergence_signal'].rolling(window=5, min_periods=1).sum()
    
    # Price Momentum Framework
    data['hl_avg'] = (data['high'] + data['low']) / 2
    data['ultra_short_momentum'] = data['hl_avg'] - data['hl_avg'].shift(1)
    data['short_term_momentum'] = data['close'] - data['close'].shift(3)
    data['medium_term_momentum'] = data['close'] - data['close'].shift(8)
    
    # Volume Momentum Analysis
    data['volume_acceleration'] = data['volume'] - data['volume'].shift(1)
    data['volume_trend'] = data['volume'] - data['volume'].shift(4)
    
    # Volume persistence (consecutive same-direction changes)
    def volume_persistence(series, window=3):
        signs = np.sign(series.diff())
        persistence = signs.rolling(window=window).apply(
            lambda x: len(set(x.dropna())) == 1 if not x.isna().all() else 0, raw=False
        )
        return persistence
    
    data['volume_persistence'] = volume_persistence(data['volume'])
    
    # Range-Based Analysis
    data['daily_range'] = data['high'] - data['low']
    data['range_expansion'] = data['daily_range'] - data['daily_range'].shift(1)
    data['range_position'] = (data['close'] - data['low']) / (data['daily_range'] + 1e-8)
    
    # Volume-Range Integration
    data['volume_per_range'] = data['volume'] / (data['daily_range'] + 1e-8)
    data['range_volume_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['volume_per_range'] + 1e-8)
    
    # Divergence-Momentum Integration
    data['momentum_divergence'] = data['ultra_short_momentum'] * np.sign(data['volume_acceleration'])
    data['divergence_acceleration'] = data['short_term_div'] - data['medium_term_div']
    data['volume_price_alignment'] = np.sign(data['volume_trend']) * np.sign(data['short_term_momentum'])
    
    # Range-Divergence Relationships
    data['range_divergence_corr'] = data['range_expansion'] * data['divergence_intensity']
    data['position_divergence_align'] = data['range_position'] * data['divergence_signal']
    
    # Breakout divergence
    data['breakout_divergence'] = ((data['high'] > data['high'].shift(1)) & (data['divergence_signal'] == -1)).astype(int)
    
    # Concentration divergence
    data['volume_per_range_trend'] = data['volume_per_range'] - data['volume_per_range'].shift(3)
    data['concentration_divergence'] = np.sign(data['volume_per_range_trend']) * np.sign(data['short_term_momentum'])
    
    # Regime Classification
    data['divergence_regime'] = data['divergence_signal'].rolling(window=5, min_periods=1).apply(
        lambda x: 2 if (x == 1).sum() >= 3 else (-2 if (x == -1).sum() >= 3 else 1 if (x == 0).sum() >= 3 else 0), 
        raw=False
    )
    
    # Dynamic Weighting System
    timeframe_weight = (0.4 * data['ultra_short_div'] + 
                       0.35 * data['short_term_div'] + 
                       0.25 * data['medium_term_div'])
    
    range_adjustment = 1 + (data['range_expansion'] / (data['daily_range'] + 1e-8))
    volume_scaling = data['volume'] / (data['volume'].shift(4) + 1e-8)
    
    # Divergence Validation
    data['momentum_divergence_consistency'] = data['short_term_momentum'] * data['divergence_signal']
    data['range_divergence_confirmation'] = data['range_position'].diff() * data['divergence_intensity']
    data['volume_concentration_alignment'] = data['volume_per_range'] * abs(data['divergence_intensity'])
    
    # Composite Factor Construction
    base_divergence = timeframe_weight * range_adjustment * volume_scaling
    
    # Signal enhancement components
    momentum_confirmation = data['momentum_divergence_consistency'] * 0.3
    range_confirmation = data['range_divergence_confirmation'] * 0.25
    volume_alignment = data['volume_concentration_alignment'] * 0.2
    divergence_strength = data['divergence_intensity'] * 0.25
    
    # Final composite factor
    composite_factor = (base_divergence + 
                       momentum_confirmation + 
                       range_confirmation + 
                       volume_alignment + 
                       divergence_strength)
    
    # Signal quality adjustment
    divergence_consistency = (abs(data['ultra_short_div']) + 
                             abs(data['short_term_div']) + 
                             abs(data['medium_term_div'])) / 3
    
    volume_range_alignment = 1 - abs(data['volume_price_alignment'] - np.sign(composite_factor))
    
    signal_quality = divergence_consistency * volume_range_alignment
    
    final_factor = composite_factor * signal_quality
    
    return final_factor
