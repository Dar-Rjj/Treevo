import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    
    # Price Momentum Components
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_momentum'] = (data['open'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    data['close_momentum'] = (data['close'] - data['prev_close']) / data['prev_close'].replace(0, np.nan)
    
    # Volume Momentum Components
    data['volume_change'] = (data['volume'] - data['prev_volume']) / data['prev_volume'].replace(0, np.nan)
    data['volume_alignment'] = np.sign(data['volume'] - data['prev_volume']) * np.sign(data['close'] - data['prev_close'])
    
    # Range Momentum Components
    data['range_current'] = data['high'] - data['low']
    data['range_previous'] = data['prev_high'] - data['prev_low']
    data['range_expansion'] = data['range_current'] / data['range_previous'].replace(0, np.nan)
    data['position_current'] = (data['close'] - data['low']) / data['range_current'].replace(0, np.nan)
    data['position_previous'] = (data['prev_close'] - data['prev_low']) / data['range_previous'].replace(0, np.nan)
    data['position_momentum'] = data['position_current'] - data['position_previous']
    
    # Divergence Analysis
    data['price_volume_div'] = np.sign(data['close'] - data['prev_close']) * np.sign(data['volume'] - data['prev_volume'])
    data['intraday_range_div'] = np.sign(data['close'] - data['open']) * np.sign(data['range_current'] - data['range_previous'])
    data['gap_close_div'] = np.sign(data['open'] - data['prev_close']) * np.sign(data['close'] - data['open'])
    
    # Consistency Scoring - Directional Agreement Count
    momentum_signs = pd.DataFrame({
        'intraday': np.sign(data['intraday_momentum']),
        'gap': np.sign(data['gap_momentum']),
        'close': np.sign(data['close_momentum']),
        'volume': np.sign(data['volume_change'])
    })
    data['directional_agreement'] = momentum_signs.apply(lambda x: (x == x.mode()[0]).sum() if len(x.mode()) > 0 else 0, axis=1)
    
    # Market Context Integration
    data['volatility_abs'] = data['range_current']
    data['volatility_rel'] = data['range_current'] / data['range_previous'].replace(0, np.nan)
    
    # Trend Context
    data['short_trend'] = data['close'].rolling(window=3, min_periods=1).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0)
    data['medium_trend'] = data['close'].rolling(window=7, min_periods=1).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0)
    
    # Fragmentation Score Calculation
    divergence_signals = pd.DataFrame({
        'price_volume': data['price_volume_div'],
        'intraday_range': data['intraday_range_div'],
        'gap_close': data['gap_close_div']
    })
    
    # Weighted fragmentation score (negative values indicate divergence)
    data['fragmentation_score'] = (
        divergence_signals['price_volume'] * 0.4 +
        divergence_signals['intraday_range'] * 0.35 +
        divergence_signals['gap_close'] * 0.25
    )
    
    # Signal Generation based on fragmentation
    data['base_signal'] = np.where(
        data['fragmentation_score'] < -0.5,  # High fragmentation -> mean reversion
        -data['close_momentum'],  # Bet against recent momentum
        data['close_momentum']   # Low fragmentation -> momentum continuation
    )
    
    # Context Adjustment
    volatility_scale = 1 / (1 + data['volatility_rel'].abs())
    trend_alignment = np.where(
        data['short_trend'] == data['medium_trend'], 
        1.2,  # Strong trend alignment
        0.8   # Weak trend alignment
    )
    
    # Final factor construction
    data['alpha_factor'] = (
        data['base_signal'] * 
        volatility_scale * 
        trend_alignment * 
        (1 + 0.1 * data['directional_agreement'])
    )
    
    # Clean up intermediate columns and return final factor
    result = data['alpha_factor'].copy()
    return result
