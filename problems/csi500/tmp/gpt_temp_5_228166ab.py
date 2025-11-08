import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-timeframe momentum convergence factor with regime-based signal integration
    and volume anomaly enhancement for stock return prediction.
    """
    
    # Calculate basic price components
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    amount = df['amount']
    
    # Initialize result series
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    # Multi-Timeframe Momentum Components
    # Short-term (3-day) momentum
    roc_3d = close.pct_change(3)
    hl_range_3d = (high.rolling(3).max() - low.rolling(3).min()) / close.shift(3)
    open_close_mom_3d = (close - open_price.shift(3)) / open_price.shift(3)
    vol_roc_3d = volume.pct_change(3)
    vol_price_corr_3d = close.rolling(3).corr(volume)
    vol_accel_3d = volume.pct_change(3) - volume.pct_change(6)
    
    short_term_momentum = (
        0.4 * roc_3d + 
        0.25 * hl_range_3d + 
        0.2 * open_close_mom_3d + 
        0.15 * (0.4 * vol_roc_3d + 0.35 * vol_price_corr_3d + 0.25 * vol_accel_3d)
    )
    
    # Medium-term (10-day) momentum
    roc_10d = close.pct_change(10)
    true_range = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    tr_trend_10d = true_range.rolling(10).mean() / true_range.rolling(20).mean()
    volatility_adj_mom_10d = roc_10d / close.rolling(10).std()
    vol_roc_10d = volume.pct_change(10)
    vol_price_corr_10d = close.rolling(10).corr(volume)
    vol_persistence_10d = (volume.rolling(5).mean() > volume.rolling(10).mean()).astype(float)
    
    medium_term_momentum = (
        0.35 * roc_10d + 
        0.25 * tr_trend_10d + 
        0.2 * volatility_adj_mom_10d + 
        0.2 * (0.4 * vol_roc_10d + 0.35 * vol_price_corr_10d + 0.25 * vol_persistence_10d)
    )
    
    # Long-term (20-day) momentum
    roc_20d = close.pct_change(20)
    price_trend_consistency = close.rolling(10).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1], raw=True)
    vol_regime_adj_mom_20d = roc_20d / close.rolling(20).std()
    vol_roc_20d = volume.pct_change(20)
    vol_price_corr_20d = close.rolling(20).corr(volume)
    vol_trend_stability = volume.rolling(10).std() / volume.rolling(20).mean()
    
    long_term_momentum = (
        0.4 * roc_20d + 
        0.25 * price_trend_consistency + 
        0.2 * vol_regime_adj_mom_20d + 
        0.15 * (0.4 * vol_roc_20d + 0.35 * vol_price_corr_20d + 0.25 * (1 - vol_trend_stability))
    )
    
    # Market Regime Detection
    # Volatility regime
    atr_14d = true_range.rolling(14).mean()
    atr_median_20d = atr_14d.rolling(20).median()
    atr_ratio = atr_14d / atr_median_20d
    
    volatility_regime = pd.Series('normal', index=df.index)
    volatility_regime[atr_ratio < 0.6] = 'low'
    volatility_regime[atr_ratio > 1.4] = 'high'
    
    # Price range regime
    hl_range = high - low
    avg_range_20d = hl_range.rolling(20).mean()
    range_ratio = hl_range / avg_range_20d
    
    range_regime = pd.Series('normal', index=df.index)
    range_regime[range_ratio < 0.7] = 'compressed'
    range_regime[range_ratio > 1.3] = 'expanded'
    
    # Volume regime
    vol_avg_20d = volume.rolling(20).mean()
    vol_ratio = volume / vol_avg_20d
    
    volume_level_regime = pd.Series('normal', index=df.index)
    volume_level_regime[vol_ratio < 0.7] = 'low'
    volume_level_regime[vol_ratio > 1.5] = 'high'
    
    # Volume trend regime
    vol_trend_5d = volume.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    vol_trend_pct = vol_trend_5d / volume.rolling(5).mean()
    
    volume_trend_regime = pd.Series('stable', index=df.index)
    volume_trend_regime[vol_trend_pct < -0.1] = 'declining'
    volume_trend_regime[vol_trend_pct > 0.1] = 'rising'
    
    # Regime-Adaptive Signal Weighting
    regime_weights = pd.DataFrame(index=df.index, columns=['short', 'medium', 'long', 'volume_mult'])
    
    for i in df.index:
        vol_regime = volatility_regime.loc[i]
        vol_level = volume_level_regime.loc[i]
        vol_trend = volume_trend_regime.loc[i]
        
        if vol_regime == 'low' and vol_level in ['normal', 'high']:
            regime_weights.loc[i, ['short', 'medium', 'long']] = [0.5, 0.3, 0.2]
            regime_weights.loc[i, 'volume_mult'] = 1.2
        elif vol_regime == 'high' and vol_level in ['normal', 'high']:
            regime_weights.loc[i, ['short', 'medium', 'long']] = [0.2, 0.35, 0.45]
            regime_weights.loc[i, 'volume_mult'] = 1.1
        elif vol_regime == 'normal' and vol_level == 'high':
            regime_weights.loc[i, ['short', 'medium', 'long']] = [0.35, 0.35, 0.2]
            regime_weights.loc[i, 'volume_mult'] = 1.3
        elif vol_level == 'low':
            regime_weights.loc[i, ['short', 'medium', 'long']] = [0.25, 0.25, 0.15]
            regime_weights.loc[i, 'volume_mult'] = 0.5
        else:
            regime_weights.loc[i, ['short', 'medium', 'long']] = [0.3, 0.3, 0.25]
            regime_weights.loc[i, 'volume_mult'] = 1.0
    
    # Volume Anomaly Detection
    # Extreme volume events
    breakout_volume = (vol_ratio > 2.5) & (close > close.rolling(5).max())
    capitulation_volume = (vol_ratio > 3.0) & (close < close.rolling(5).min())
    accumulation_volume = (vol_ratio > 2.0) & (hl_range / close < 0.02)
    
    # Volume dry-ups
    extreme_low_volume = vol_ratio < 0.3
    declining_volume_trend = (volume.pct_change(5) < -0.2) & (volume.pct_change(4) < -0.2)
    volume_divergence = (close.pct_change(5) > 0.05) & (volume.pct_change(5) < -0.1)
    
    # Volume pattern recognition
    volume_clustering = (vol_ratio > 1.5).rolling(3).sum() >= 2
    volume_expansion = (vol_trend_pct > 0.05).rolling(3).sum() >= 2
    volume_contraction = (vol_trend_pct < -0.05).rolling(3).sum() >= 2
    
    # Volume anomaly multiplier
    volume_multiplier = pd.Series(1.0, index=df.index)
    
    # Apply volume multipliers based on anomalies
    volume_multiplier[breakout_volume] = 1.8
    volume_multiplier[capitulation_volume] = 1.5
    volume_multiplier[accumulation_volume] = 1.3
    volume_multiplier[extreme_low_volume] = 0.3
    volume_multiplier[declining_volume_trend] = 0.7
    volume_multiplier[volume_divergence] = 0.8
    volume_multiplier[volume_clustering] = 1.4
    volume_multiplier[volume_expansion] = 1.6
    volume_multiplier[volume_contraction] = 0.6
    
    # Cap volume multiplier
    volume_multiplier = volume_multiplier.clip(0.3, 2.0)
    
    # Calculate final alpha factor
    for i in df.index:
        if pd.notna(short_term_momentum.loc[i]) and pd.notna(medium_term_momentum.loc[i]) and pd.notna(long_term_momentum.loc[i]):
            weights = regime_weights.loc[i, ['short', 'medium', 'long']]
            base_score = (
                weights['short'] * short_term_momentum.loc[i] +
                weights['medium'] * medium_term_momentum.loc[i] +
                weights['long'] * long_term_momentum.loc[i]
            )
            
            regime_mult = regime_weights.loc[i, 'volume_mult']
            vol_anomaly_mult = volume_multiplier.loc[i]
            
            alpha_factor.loc[i] = base_score * regime_mult * vol_anomaly_mult
    
    # Normalize the final factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(20).mean()) / alpha_factor.rolling(20).std()
    
    return alpha_factor
