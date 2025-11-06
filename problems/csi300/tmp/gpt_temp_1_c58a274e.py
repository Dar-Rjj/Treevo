import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Divergence with Regime-Adaptive Weighting
    """
    data = df.copy()
    
    # Price-Volume Divergence Framework
    # Directional Volume Analysis
    data['prev_close'] = data['close'].shift(1)
    data['up_volume'] = np.where(data['close'] > data['prev_close'], data['volume'], 0)
    data['down_volume'] = np.where(data['close'] < data['prev_close'], data['volume'], 0)
    
    # Rolling directional volume ratios
    data['up_volume_ma'] = data['up_volume'].rolling(window=5, min_periods=3).mean()
    data['down_volume_ma'] = data['down_volume'].rolling(window=5, min_periods=3).mean()
    data['directional_volume_ratio'] = data['up_volume_ma'] / (data['down_volume_ma'] + 1e-8)
    
    # Price Momentum Divergence
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['return_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_divergence'] = data['return_3d'] - data['return_8d']
    
    # Market Regime Detection System
    # Volatility Regime Classification
    data['daily_range'] = data['high'] - data['low']
    data['volatility_5d'] = data['daily_range'].rolling(window=5, min_periods=3).std()
    
    # Calculate volatility percentiles for regime classification
    vol_percentiles = data['volatility_5d'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    data['volatility_regime'] = np.select(
        [vol_percentiles < 0.33, vol_percentiles > 0.67],
        [0, 2],  # 0=low vol, 1=medium vol, 2=high vol
        default=1
    )
    
    # Trend Regime Identification
    def calculate_slope(window):
        if len(window) < 5:
            return np.nan
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window)
        return slope
    
    data['price_slope_10d'] = data['close'].rolling(window=10, min_periods=5).apply(
        calculate_slope, raw=True
    )
    data['trend_strength'] = data['price_slope_10d'].abs()
    
    # Adaptive Signal Weighting
    # Regime-Dependent Scaling
    regime_weights = {
        0: 0.7,  # Low volatility - conservative weighting
        1: 1.0,  # Medium volatility - normal weighting
        2: 1.3   # High volatility - aggressive weighting
    }
    data['volatility_weight'] = data['volatility_regime'].map(regime_weights)
    
    # Trend-based adjustment
    trend_threshold = data['trend_strength'].rolling(window=20, min_periods=10).quantile(0.6)
    data['trend_weight'] = np.where(
        data['trend_strength'] > trend_threshold, 1.2, 0.8
    )
    
    # Volume Confirmation Logic
    data['volume_persistence'] = (
        data['directional_volume_ratio'].rolling(window=3, min_periods=2).std()
    )
    data['volume_confidence'] = 1 / (data['volume_persistence'] + 1e-8)
    
    # Composite Alpha Generation
    # Divergence Signal Construction
    data['raw_divergence_signal'] = (
        data['momentum_divergence'] * 
        np.sign(data['directional_volume_ratio'] - 1) *
        data['volume_confidence']
    )
    
    # Apply regime-adaptive scaling
    data['scaled_divergence'] = (
        data['raw_divergence_signal'] * 
        data['volatility_weight'] * 
        data['trend_weight']
    )
    
    # Multi-Timeframe Integration
    # Blend short-term and medium-term signals
    short_term_signal = data['scaled_divergence'].rolling(window=3, min_periods=2).mean()
    medium_term_signal = data['scaled_divergence'].rolling(window=8, min_periods=5).mean()
    
    # Dynamic weighting based on regime transitions
    regime_changes = data['volatility_regime'].diff().abs()
    regime_volatility = np.where(regime_changes > 0, 0.7, 0.3)  # Reduce weight during transitions
    
    # Final composite alpha
    alpha = (
        regime_volatility * short_term_signal + 
        (1 - regime_volatility) * medium_term_signal
    )
    
    return alpha
