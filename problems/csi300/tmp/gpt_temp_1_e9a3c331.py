import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume divergence, volatility regime breakout quality,
    liquidity-confirmed mean reversion, and trend-volume convergence signals.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Price-Volume Divergence Momentum
    # Mid-range momentum calculation
    mid_range = (data['high'] + data['low']) / 2
    data['mid_range_5d_return'] = mid_range.pct_change(5)
    data['mid_range_10d_return'] = mid_range.pct_change(10)
    
    # Volume momentum calculation
    data['volume_5d_return'] = data['volume'].pct_change(5)
    data['volume_10d_return'] = data['volume'].pct_change(10)
    
    # Divergence signal generation
    data['short_term_divergence'] = data['mid_range_5d_return'] - data['volume_5d_return']
    data['medium_term_divergence'] = data['mid_range_10d_return'] - data['volume_10d_return']
    
    # Volatility-Regime Breakout Quality
    # True range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Efficiency ratio: Close movement vs true range
    data['close_movement'] = abs(data['close'] - data['close'].shift(5))
    data['true_range_5d'] = data['true_range'].rolling(window=5).sum()
    data['efficiency_ratio'] = data['close_movement'] / data['true_range_5d']
    
    # Volatility regime detection
    data['return_volatility_20d'] = data['close'].pct_change().rolling(window=20).std()
    volatility_median = data['return_volatility_20d'].rolling(window=100).median()
    data['high_vol_regime'] = (data['return_volatility_20d'] > volatility_median).astype(int)
    
    # Regime-adaptive signals
    data['breakout_quality'] = np.where(
        data['high_vol_regime'] == 1,
        -data['efficiency_ratio'],  # High volatility + efficient breakout → reversal
        data['efficiency_ratio']    # Low volatility + inefficient breakout → fade
    )
    
    # Liquidity-Confirmed Mean Reversion
    # Price deviation setup
    data['mid_range_ma_20d'] = mid_range.rolling(window=20).mean()
    data['mid_range_deviation'] = (mid_range - data['mid_range_ma_20d']) / data['mid_range_ma_20d']
    data['opening_gap'] = (data['open'] - mid_range.shift(1)) / mid_range.shift(1)
    
    # Volume confirmation
    data['volume_volatility_20d'] = data['volume'].pct_change().rolling(window=20).std()
    data['volume_momentum_divergence'] = data['volume_5d_return'] - data['volume_10d_return']
    
    # Mean reversion signal with volume confirmation
    data['liquidity_reversion'] = data['mid_range_deviation'] * np.where(
        data['volume_volatility_20d'] > data['volume_volatility_20d'].rolling(window=50).median(),
        -data['volume_momentum_divergence'],  # High volume volatility
        data['volume_momentum_divergence']    # Normal volume volatility
    )
    
    # Trend-Volume Convergence
    # Mid-range trend analysis
    data['mid_range_trend_10d'] = mid_range.rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) == 10 else np.nan
    )
    
    # Directional persistence
    data['mid_range_direction_5d'] = mid_range.diff(5).apply(np.sign)
    data['direction_persistence'] = data['mid_range_direction_5d'].rolling(window=15).apply(
        lambda x: x.sum() / len(x) if len(x) == 15 else np.nan
    )
    
    # Volume pattern integration
    data['volume_clustering'] = data['volume'].rolling(window=10).apply(
        lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan
    )
    
    # Volume breakout confirmation
    data['volume_breakout'] = (data['volume'] > data['volume'].rolling(window=20).mean() * 1.5).astype(int)
    
    # Trend-volume convergence signal
    data['trend_volume_convergence'] = data['mid_range_trend_10d'] * data['direction_persistence'] * data['volume_breakout']
    
    # Combine all signals into final alpha factor
    # Normalize components before combination
    components = ['short_term_divergence', 'medium_term_divergence', 'breakout_quality', 
                 'liquidity_reversion', 'trend_volume_convergence']
    
    for component in components:
        if component in data.columns:
            data[f'{component}_norm'] = (data[component] - data[component].rolling(window=50).mean()) / data[component].rolling(window=50).std()
    
    # Final alpha factor - weighted combination of normalized components
    alpha = (
        0.25 * data.get('short_term_divergence_norm', 0) +
        0.20 * data.get('medium_term_divergence_norm', 0) +
        0.25 * data.get('breakout_quality_norm', 0) +
        0.15 * data.get('liquidity_reversion_norm', 0) +
        0.15 * data.get('trend_volume_convergence_norm', 0)
    )
    
    return alpha
