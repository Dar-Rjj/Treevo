import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Adaptive Momentum with Volume-Price Integration
    Generates a composite alpha factor combining momentum across multiple timeframes
    with volume confirmation and volatility-adaptive processing.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate basic price features
    data['prev_close'] = data['close'].shift(1)
    data['prev_high'] = data['high'].shift(1)
    
    # Multi-Scale Momentum Framework
    # Short-Term Momentum (3-5 days)
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['short_term_accel'] = data['momentum_5d'] - data['momentum_3d']
    
    # Medium-Term Momentum (10-15 days)
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_15d'] = data['close'] / data['close'].shift(15) - 1
    data['medium_term_consistency'] = data['momentum_15d'] - data['momentum_10d']
    
    # Long-Term Momentum (20-30 days)
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_30d'] = data['close'] / data['close'].shift(30) - 1
    data['long_term_trend_strength'] = data['momentum_30d'] - data['momentum_20d']
    
    # Volume-Price Integration System
    # Volume moving averages
    data['volume_ma_5'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_15'] = data['volume'].rolling(window=15).mean()
    data['volume_ma_20'] = data['volume'].rolling(window=20).mean()
    
    # Volume Momentum Analysis
    data['volume_momentum_5d'] = data['volume'] / data['volume_ma_5'] - 1
    data['volume_momentum_15d'] = data['volume'] / data['volume_ma_15'] - 1
    data['volume_acceleration'] = data['volume_momentum_5d'] - data['volume_momentum_15d']
    
    # Price-Volume Divergence Detection
    data['bullish_divergence'] = ((data['momentum_5d'] > 0) & (data['volume_momentum_5d'] < 0)).astype(int)
    data['bearish_divergence'] = ((data['momentum_5d'] < 0) & (data['volume_momentum_5d'] > 0)).astype(int)
    data['divergence_strength'] = abs(data['momentum_5d']) * abs(data['volume_momentum_5d'])
    
    # Volume Breakout Confirmation
    data['volume_surge'] = (data['volume'] > data['volume_ma_20'] * 2.0).astype(int)
    data['price_confirmation'] = (data['close'] > data['prev_high']).astype(int)
    data['breakout_strength'] = (data['close'] - data['prev_high']) * data['volume']
    
    # Volatility-Adaptive Signal Processing
    # ATR calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['atr_20'] = data['tr'].rolling(window=20).mean()
    
    # Volatility Environment Classification
    atr_60d_percentile_75 = data['atr_20'].rolling(window=60).quantile(0.75)
    atr_60d_percentile_25 = data['atr_20'].rolling(window=60).quantile(0.25)
    
    data['high_vol_regime'] = (data['atr_20'] > atr_60d_percentile_75).astype(int)
    data['low_vol_regime'] = (data['atr_20'] < atr_60d_percentile_25).astype(int)
    data['normal_vol_regime'] = ((~data['high_vol_regime'].astype(bool)) & 
                                (~data['low_vol_regime'].astype(bool))).astype(int)
    
    # Price Action Quality Assessment
    # Trend Quality Metrics
    data['trend_consistency'] = data['close'].rolling(window=10).apply(
        lambda x: (x > x.shift(1)).sum() / 10, raw=False
    )
    
    # ATR for volatility efficiency
    data['atr_10d'] = data['tr'].rolling(window=10).mean()
    data['volatility_efficiency'] = (data['close'] - data['close'].shift(10)) / data['atr_10d'].rolling(window=10).sum()
    data['price_stability'] = 1 / (data['high'] - data['low'])
    
    # Support/Resistance Analysis
    data['high_20d'] = data['high'].rolling(window=20).max()
    data['low_20d'] = data['low'].rolling(window=20).min()
    data['distance_to_high'] = (data['high_20d'] - data['close']) / (data['high_20d'] - data['low_20d'])
    data['distance_to_low'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    data['breakout_potential'] = np.minimum(data['distance_to_high'], data['distance_to_low'])
    
    # Market Microstructure Signals
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['atr_5'] = data['tr'].rolling(window=5).mean()
    data['overnight_momentum'] = (data['open'] - data['prev_close']) / data['atr_5']
    data['session_dominance'] = abs(data['close'] - data['open']) / data['tr']
    
    # Alpha Factor Synthesis Engine
    # Multi-Dimensional Signal Integration
    # Volume alignment score
    data['volume_alignment_score'] = np.where(
        data['momentum_5d'] * data['volume_momentum_5d'] > 0,
        abs(data['volume_momentum_5d']),
        -abs(data['volume_momentum_5d'])
    )
    
    # Quality adjustment
    data['trend_quality'] = data['trend_consistency'] * data['volatility_efficiency']
    data['quality_adjustment'] = data['trend_quality'] * data['price_stability']
    
    # Regime-Adaptive Final Processing
    # Initialize composite factor
    data['composite_factor'] = 0.0
    
    # High volatility regime: emphasize medium-term momentum
    high_vol_mask = data['high_vol_regime'] == 1
    data.loc[high_vol_mask, 'composite_factor'] = (
        0.6 * data['momentum_15d'] + 
        0.3 * data['momentum_5d'] + 
        0.1 * data['momentum_30d']
    )
    
    # Low volatility regime: emphasize short-term momentum
    low_vol_mask = data['low_vol_regime'] == 1
    data.loc[low_vol_mask, 'composite_factor'] = (
        0.7 * data['momentum_5d'] + 
        0.2 * data['momentum_15d'] + 
        0.1 * data['momentum_30d']
    )
    
    # Normal volatility regime: balanced weighting
    normal_vol_mask = data['normal_vol_regime'] == 1
    data.loc[normal_vol_mask, 'composite_factor'] = (
        0.4 * data['momentum_5d'] + 
        0.4 * data['momentum_15d'] + 
        0.2 * data['momentum_30d']
    )
    
    # Apply volume confirmation multiplier
    data['volume_multiplier'] = 1 + data['volume_alignment_score']
    data['composite_factor'] *= data['volume_multiplier']
    
    # Apply quality adjustment
    data['composite_factor'] *= data['quality_adjustment']
    
    # Adjust for market microstructure
    data['composite_factor'] *= (1 + data['intraday_strength'])
    
    # Incorporate support/resistance positioning
    resistance_adjustment = np.where(
        data['distance_to_high'] < 0.1,  # Near resistance
        -0.2,
        np.where(
            data['distance_to_low'] < 0.1,  # Near support
            0.1,
            0.0  # Neutral zone
        )
    )
    data['composite_factor'] *= (1 + resistance_adjustment)
    
    # Final factor value
    factor = data['composite_factor']
    
    # Clean up: remove intermediate columns and handle NaN values
    factor = factor.replace([np.inf, -np.inf], np.nan)
    
    return factor
