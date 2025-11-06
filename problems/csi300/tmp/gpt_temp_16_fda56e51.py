import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Fractal Break-Momentum Integration Alpha Factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize output series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # Calculate basic components
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # ATR calculation (5-day)
    data['tr'] = np.maximum(data['high'] - data['low'], 
                           np.maximum(abs(data['high'] - data['prev_close']), 
                                     abs(data['low'] - data['prev_close'])))
    data['atr_5'] = data['tr'].rolling(window=5, min_periods=5).mean()
    
    # Multi-Timeframe Break Detection
    for i in range(len(data)):
        if i < 8:  # Need enough history
            alpha.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        # Short-term break detection (3-day lookback)
        short_term_high = max(data['high'].iloc[i-3:i])
        short_term_low = min(data['low'].iloc[i-3:i])
        short_break_up = current['close'] > short_term_high
        short_break_down = current['close'] < short_term_low
        
        # Medium-term break detection (8-day lookback)
        medium_term_high = max(data['high'].iloc[i-8:i])
        medium_term_low = min(data['low'].iloc[i-8:i])
        medium_break_up = current['close'] > medium_term_high
        medium_break_down = current['close'] < medium_term_low
        
        # Break significance calculation
        if short_break_up or medium_break_up:
            ref_level = short_term_high if short_break_up else medium_term_high
            break_significance = abs(current['close'] - ref_level) / (current['high'] - current['low'])
            break_direction = 1
        elif short_break_down or medium_break_down:
            ref_level = short_term_low if short_break_down else medium_term_low
            break_significance = abs(current['close'] - ref_level) / (current['high'] - current['low'])
            break_direction = -1
        else:
            break_significance = 0
            break_direction = 0
        
        # Volume confirmation
        vol_median = np.median(data['volume'].iloc[i-5:i])
        volume_confirmation = current['volume'] / vol_median if vol_median > 0 else 1
        
        # Break cleanliness
        break_cleanliness = (current['high'] - current['low']) / current['atr_5'] if current['atr_5'] > 0 else 1
        
        # Gap-Fractal Integration
        gap_size = abs(current['open'] - data['close'].iloc[i-1]) / (data['high'].iloc[i-1] - data['low'].iloc[i-1]) if (data['high'].iloc[i-1] - data['low'].iloc[i-1]) > 0 else 0
        gap_direction = np.sign(current['open'] - data['close'].iloc[i-1])
        intraday_direction = np.sign(current['close'] - current['open'])
        gap_persistence_quality = gap_direction * intraday_direction * break_significance
        
        # Multi-Scale Fractal Momentum
        # Intraday fractal momentum
        intraday_fractal_momentum = (current['close'] - current['open']) / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        
        # Short-term fractal momentum (2-day)
        price_change_short = current['close'] - data['close'].iloc[i-2]
        range_sum_short = sum(data['high'].iloc[j] - data['low'].iloc[j] for j in range(i-2, i+1))
        short_term_fractal_momentum = price_change_short / range_sum_short if range_sum_short > 0 else 0
        
        # Medium-term fractal momentum (5-day)
        price_change_medium = current['close'] - data['close'].iloc[i-5]
        avg_range_medium = np.mean([data['high'].iloc[j] - data['low'].iloc[j] for j in range(i-5, i+1)])
        medium_term_fractal_momentum = price_change_medium / avg_range_medium if avg_range_medium > 0 else 0
        
        # Fractal Volatility Context
        short_term_fractal_vol = (current['high'] - current['low']) / current['close'] if current['close'] > 0 else 0
        avg_range_5 = np.mean([data['high'].iloc[j] - data['low'].iloc[j] for j in range(i-5, i+1)])
        fractal_vol_ratio = (current['high'] - current['low']) / avg_range_5 if avg_range_5 > 0 else 1
        
        # Volatility regime weighting
        if fractal_vol_ratio > 1.5:
            volatility_weight = 0.4
        elif fractal_vol_ratio < 0.6:
            volatility_weight = 1.6
        else:
            volatility_weight = 1.0
        
        # Microstructure Analysis
        trade_size_efficiency = current['volume'] / current['amount'] if current['amount'] > 0 else 0
        price_impact_ratio = abs(current['close'] - current['open']) / (current['amount'] / current['volume']) if (current['amount'] / current['volume']) > 0 else 0
        impact_quality = price_impact_ratio / (current['high'] - current['low']) if (current['high'] - current['low']) > 0 else 0
        
        # Microstructure break filters
        microstructure_adjustment = 1.0
        if break_direction != 0:
            if impact_quality > 0.7:
                microstructure_adjustment = 0.7  # Reduce signal 30%
            elif impact_quality < 0.3:
                microstructure_adjustment = 1.4  # Enhance signal 40%
        
        # Core Factor Construction
        # Primary: Volatility-weighted break momentum
        volatility_weighted_break_momentum = break_significance * short_term_fractal_momentum * volatility_weight
        
        # Confirmation: Fractal break confirmation
        fractal_divergence_momentum = short_term_fractal_momentum * break_cleanliness
        fractal_break_confirmation = break_direction * fractal_divergence_momentum * volume_confirmation
        
        # Enhancement: Gap-break-momentum alignment
        gap_break_momentum_alignment = gap_persistence_quality * break_significance * intraday_fractal_momentum
        
        # Composite Alpha Assembly
        core_factor = (volatility_weighted_break_momentum + 
                      fractal_break_confirmation + 
                      gap_break_momentum_alignment) * microstructure_adjustment
        
        alpha.iloc[i] = core_factor
    
    # Fill initial NaN values with 0
    alpha = alpha.fillna(0)
    
    return alpha
