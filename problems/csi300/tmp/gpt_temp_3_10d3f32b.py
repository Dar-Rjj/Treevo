import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Price-Volume Asymmetry with Structural Regime Detection
    """
    data = df.copy()
    
    # Price-Volume Asymmetry Analysis
    # Directional volume asymmetry
    data['returns'] = data['close'].pct_change()
    data['up_day'] = (data['returns'] > 0).astype(int)
    data['down_day'] = (data['returns'] < 0).astype(int)
    
    # Rolling volume asymmetry
    window_short = 5
    window_medium = 21
    window_long = 63
    
    # Up-day vs Down-day volume ratio
    data['up_volume_short'] = data['volume'].rolling(window_short).apply(
        lambda x: x[data['up_day'].iloc[-len(x):].values.astype(bool)].mean() if any(data['up_day'].iloc[-len(x):]) else 1, 
        raw=False
    )
    data['down_volume_short'] = data['volume'].rolling(window_short).apply(
        lambda x: x[data['down_day'].iloc[-len(x):].values.astype(bool)].mean() if any(data['down_day'].iloc[-len(x):]) else 1, 
        raw=False
    )
    data['volume_asymmetry_short'] = data['up_volume_short'] / data['down_volume_short']
    
    # Volume concentration on extreme moves
    data['price_range'] = (data['high'] - data['low']) / data['close']
    data['extreme_move'] = (data['price_range'] > data['price_range'].rolling(20).quantile(0.8)).astype(int)
    data['extreme_volume_ratio'] = data['volume'].rolling(10).apply(
        lambda x: x[data['extreme_move'].iloc[-len(x):].values.astype(bool)].mean() / x.mean() if len(x) > 0 else 1,
        raw=False
    )
    
    # Price-volume divergence patterns
    data['price_momentum_short'] = data['close'].pct_change(3)
    data['volume_momentum_short'] = data['volume'].pct_change(3)
    data['pv_divergence'] = (data['price_momentum_short'] - data['volume_momentum_short']) / (
        abs(data['price_momentum_short']) + abs(data['volume_momentum_short']) + 1e-8
    )
    
    # Structural Regime Detection
    # Market microstructure regime
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['range_efficiency'] = data['daily_range'] / (data['volume'].rolling(5).std() + 1e-8)
    
    # Trend-structure classification
    data['trend_strength'] = data['close'].rolling(10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.std() + 1e-8), 
        raw=False
    )
    data['mean_reversion_strength'] = -abs(data['trend_strength'])
    
    # Regime classification
    data['trending_regime'] = (abs(data['trend_strength']) > 1.0).astype(int)
    data['mean_reverting_regime'] = (abs(data['trend_strength']) < 0.5).astype(int)
    data['transition_regime'] = ((abs(data['trend_strength']) >= 0.5) & (abs(data['trend_strength']) <= 1.0)).astype(int)
    
    # Multi-Scale Momentum Persistence
    # Momentum across timeframes
    mom_short = data['close'].pct_change(3)
    mom_medium = data['close'].pct_change(10)
    mom_long = data['close'].pct_change(21)
    
    data['momentum_consistency'] = (
        np.sign(mom_short) * np.sign(mom_medium) * np.sign(mom_long) * 
        (abs(mom_short) + abs(mom_medium) + abs(mom_long)) / 3
    )
    
    # Volume-Price Efficiency Gradient
    # Volume efficiency slope
    data['volume_efficiency'] = data['returns'].abs() / (data['volume'] + 1e-8)
    data['efficiency_gradient'] = data['volume_efficiency'].diff(3)
    
    # Price efficiency under volume stress
    high_volume_threshold = data['volume'].rolling(20).quantile(0.8)
    low_volume_threshold = data['volume'].rolling(20).quantile(0.2)
    
    data['high_volume_efficiency'] = data['volume_efficiency'].rolling(5).apply(
        lambda x: x[data['volume'].iloc[-len(x):] > high_volume_threshold.iloc[-1]].mean() if any(data['volume'].iloc[-len(x):] > high_volume_threshold.iloc[-1]) else 0,
        raw=False
    )
    data['low_volume_efficiency'] = data['volume_efficiency'].rolling(5).apply(
        lambda x: x[data['volume'].iloc[-len(x):] < low_volume_threshold.iloc[-1]].mean() if any(data['volume'].iloc[-len(x):] < low_volume_threshold.iloc[-1]) else 0,
        raw=False
    )
    
    # Adaptive Signal Architecture
    # Regime-adaptive factors
    trending_factor = (
        data['volume_asymmetry_short'] * data['momentum_consistency'] * 
        data['trending_regime']
    )
    
    mean_reverting_factor = (
        data['pv_divergence'] * data['mean_reversion_strength'] * 
        data['mean_reverting_regime']
    )
    
    transition_factor = (
        data['efficiency_gradient'] * data['extreme_volume_ratio'] * 
        data['transition_regime']
    )
    
    # Multi-scale signal integration
    short_term_signal = data['volume_asymmetry_short'].rolling(3).mean()
    medium_term_signal = data['momentum_consistency'].rolling(10).mean()
    long_term_signal = data['efficiency_gradient'].rolling(21).mean()
    
    # Dynamic weighting based on regime
    regime_weights = (
        data['trending_regime'] * 0.4 + 
        data['mean_reverting_regime'] * 0.3 + 
        data['transition_regime'] * 0.3
    )
    
    # Final composite factor
    composite_factor = (
        regime_weights * (
            trending_factor.fillna(0) * 0.4 +
            mean_reverting_factor.fillna(0) * 0.35 +
            transition_factor.fillna(0) * 0.25
        ) +
        short_term_signal.fillna(0) * 0.2 +
        medium_term_signal.fillna(0) * 0.3 +
        long_term_signal.fillna(0) * 0.5
    )
    
    # Volatility adjustment
    volatility = data['returns'].rolling(20).std()
    volatility_adj = 1 / (volatility + 1e-8)
    volatility_adj = volatility_adj / volatility_adj.rolling(63).mean()
    
    final_factor = composite_factor * volatility_adj
    
    return final_factor
