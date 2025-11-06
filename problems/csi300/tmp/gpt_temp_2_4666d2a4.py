import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum Asymmetry Divergence factor
    """
    df = data.copy()
    
    # Calculate returns and true range
    df['returns'] = df['close'].pct_change()
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Define up/down days
    df['up_day'] = df['returns'] > 0
    df['down_day'] = df['returns'] < 0
    
    # Multi-Scale Asymmetry Analysis
    # Price Asymmetry Calculation
    for window in [3, 5, 10]:
        pos_returns = df['returns'].rolling(window=window).apply(
            lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else 0, raw=True
        )
        neg_returns = df['returns'].rolling(window=window).apply(
            lambda x: x[x < 0].mean() if len(x[x < 0]) > 0 else 0, raw=True
        )
        df[f'price_asym_{window}'] = pos_returns / (abs(neg_returns) + 1e-8)
    
    # Volume Asymmetry Calculation
    up_volume = df['volume'].rolling(window=5).apply(
        lambda x: x[df['up_day'].iloc[-len(x):]].mean() if len(x[df['up_day'].iloc[-len(x):]]) > 0 else 0, raw=True
    )
    down_volume = df['volume'].rolling(window=5).apply(
        lambda x: x[df['down_day'].iloc[-len(x):]].mean() if len(x[df['down_day'].iloc[-len(x):]]) > 0 else 0, raw=True
    )
    df['volume_asym'] = up_volume / (abs(down_volume) + 1e-8)
    
    # Range Efficiency Asymmetry
    efficiency_up = df.apply(
        lambda row: row['returns'] / row['true_range'] if row['up_day'] else np.nan, axis=1
    ).rolling(window=5).mean()
    efficiency_down = df.apply(
        lambda row: row['returns'] / row['true_range'] if row['down_day'] else np.nan, axis=1
    ).rolling(window=5).mean()
    df['efficiency_asym'] = efficiency_up / (abs(efficiency_down) + 1e-8)
    
    # Fractal Volatility Regime Assessment
    df['vol_short'] = (df['high'] - df['low']).rolling(window=5).mean()
    df['vol_medium'] = df['returns'].rolling(window=10).std()
    df['vol_long'] = (df['high'] - df['low']).rolling(window=20).mean()
    
    # Regime Classification
    vol_avg = (df['vol_short'] + df['vol_medium'] + df['vol_long']) / 3
    vol_std = df[['vol_short', 'vol_medium', 'vol_long']].std(axis=1)
    
    # Regime confidence based on cross-timeframe consistency
    df['regime_confidence'] = 1 / (vol_std + 1e-8)
    
    # Multi-Dimensional Asymmetry Divergence
    df['price_divergence'] = df['price_asym_3'] - df['price_asym_10']
    df['volume_divergence'] = df['volume_asym'] - df['volume_asym'].rolling(window=10).mean()
    df['efficiency_divergence'] = df['efficiency_asym'] - df['efficiency_asym'].rolling(window=10).mean()
    
    # Fractal Alignment Analysis
    df['cross_asym_corr'] = df[['price_asym_3', 'price_asym_5', 'price_asym_10']].corrwith(
        df[['volume_asym', 'efficiency_asym']].mean(axis=1), axis=0
    ).mean()
    
    # Divergence Strength Assessment
    divergence_strength = (
        abs(df['price_divergence']) + 
        abs(df['volume_divergence']) + 
        abs(df['efficiency_divergence'])
    ) / 3
    
    # Regime-Adaptive Asymmetry Integration
    # Volatility regime classification
    vol_threshold_high = vol_avg.rolling(window=20).quantile(0.7)
    vol_threshold_low = vol_avg.rolling(window=20).quantile(0.3)
    
    high_vol_regime = vol_avg > vol_threshold_high
    low_vol_regime = vol_avg < vol_threshold_low
    transition_regime = ~(high_vol_regime | low_vol_regime)
    
    # Regime-specific processing
    high_vol_factor = (
        df['efficiency_asym'].rolling(window=5).std() *  # Efficiency persistence
        df['volume_asym'] *  # Volume confirmation
        (1 / (abs(df['price_divergence']) + 1e-8)) *  # Mean reversion to price divergence
        df['regime_confidence']
    )
    
    low_vol_factor = (
        df['price_asym_5'] *  # Price asymmetry continuation
        df['efficiency_asym'] *  # Efficiency quality
        (df['volume_divergence'] > 0).astype(int) *  # Breakout detection
        df['efficiency_asym'].rolling(window=10).std()  # Range efficiency patterns
    )
    
    transition_factor = (
        (df['price_divergence'] + df['volume_divergence'] + df['efficiency_divergence']) / 3 *
        df['cross_asym_corr'] *  # Fractal alignment
        (1 - abs(df['vol_short'] - df['vol_long']) / (df['vol_short'] + df['vol_long'] + 1e-8))  # Dynamic adjustment
    )
    
    # Volume-Range Asymmetry Confirmation
    volume_clusters = (df['volume'] > df['volume'].rolling(window=20).quantile(0.7)).astype(int)
    volume_asym_alignment = df['volume_asym'] * volume_clusters
    range_utilization = df['returns'].abs() / df['true_range']
    
    asymmetry_confirmation = (
        volume_asym_alignment.rolling(window=5).mean() *
        range_utilization.rolling(window=5).mean() *
        divergence_strength
    )
    
    # Fractal Asymmetry-Divergence Convergence
    # Multi-scale pattern integration
    multi_scale_asymmetry = (
        df['price_asym_3'] * 0.4 +
        df['price_asym_5'] * 0.3 +
        df['price_asym_10'] * 0.3
    )
    
    convergence_factor = (
        multi_scale_asymmetry *
        divergence_strength *
        df['regime_confidence'] *
        asymmetry_confirmation
    )
    
    # Composite Factor Generation
    # Regime-scaled final output
    final_factor = pd.Series(index=df.index, dtype=float)
    
    # High volatility regime scaling
    high_vol_mask = high_vol_regime
    final_factor[high_vol_mask] = (
        high_vol_factor[high_vol_mask] / 
        (df['vol_short'][high_vol_mask] * df['regime_confidence'][high_vol_mask] + 1e-8)
    )
    
    # Low volatility regime scaling
    low_vol_mask = low_vol_regime
    final_factor[low_vol_mask] = (
        low_vol_factor[low_vol_mask] * 
        df['vol_short'][low_vol_mask] * 
        df['regime_confidence'][low_vol_mask]
    )
    
    # Transition regime scaling
    trans_mask = transition_regime
    transition_intensity = abs(df['vol_short'] - df['vol_long']) / (df['vol_short'] + df['vol_long'] + 1e-8)
    final_factor[trans_mask] = (
        transition_factor[trans_mask] * 
        (1 + transition_intensity[trans_mask]) *
        df['regime_confidence'][trans_mask]
    )
    
    # Incorporate volume-range asymmetry alignment
    final_factor = final_factor * (1 + asymmetry_confirmation)
    
    # Final predictive signal with pattern validation
    pattern_persistence = (
        convergence_factor.rolling(window=5).std() /
        (convergence_factor.rolling(window=20).std() + 1e-8)
    )
    
    final_signal = final_factor * pattern_persistence
    
    return final_signal
