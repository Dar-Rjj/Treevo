import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-timeframe regime-based alpha factor combining momentum, efficiency, 
    range signals with volume confirmation and regime-optimized weighting.
    """
    # Calculate returns and basic metrics
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['price_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_per_range'] = df['amount'] / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_volume_ratio'] = df['price_efficiency'] / df['volume_per_range'].replace(0, np.nan)
    
    # Volatility regime classification
    vol_20d = df['returns'].rolling(20).std()
    vol_60d = df['returns'].rolling(60).std()
    volatility_regime = pd.Series('normal', index=df.index)
    volatility_regime[(vol_20d > 1.5 * vol_60d)] = 'high'
    volatility_regime[(vol_20d < 0.67 * vol_60d)] = 'low'
    
    # Trend regime classification
    ret_20d = (df['close'] / df['close'].shift(20) - 1)
    trend_regime = pd.Series('sideways', index=df.index)
    trend_regime[ret_20d > 0.05] = 'uptrend'
    trend_regime[ret_20d < -0.05] = 'downtrend'
    
    # Volume regime classification
    vol_20d_avg = df['volume'].rolling(20).mean()
    vol_60d_avg = df['volume'].rolling(60).mean()
    volume_regime = pd.Series('normal', index=df.index)
    volume_regime[(vol_20d_avg > 1.5 * vol_60d_avg)] = 'high'
    volume_regime[(vol_20d_avg < 0.67 * vol_60d_avg)] = 'low'
    
    # Regime-optimized momentum components
    ret_5d = df['close'].pct_change(5)
    vol_5d = df['returns'].rolling(5).std()
    vol_20d_mom = df['returns'].rolling(20).std()
    
    regime_momentum = pd.Series(0.0, index=df.index)
    
    # High volatility regimes
    high_vol_mask = volatility_regime == 'high'
    uptrend_high_vol = high_vol_mask & (trend_regime == 'uptrend')
    downtrend_high_vol = high_vol_mask & (trend_regime == 'downtrend')
    
    regime_momentum[uptrend_high_vol] = ret_5d[uptrend_high_vol] / vol_5d[uptrend_high_vol].replace(0, np.nan)
    regime_momentum[downtrend_high_vol] = ret_5d[downtrend_high_vol] / vol_5d[downtrend_high_vol].replace(0, np.nan)
    
    # Low volatility regimes
    low_vol_mask = volatility_regime == 'low'
    uptrend_low_vol = low_vol_mask & (trend_regime == 'uptrend')
    downtrend_low_vol = low_vol_mask & (trend_regime == 'downtrend')
    
    regime_momentum[uptrend_low_vol] = ret_20d[uptrend_low_vol] / vol_20d_mom[uptrend_low_vol].replace(0, np.nan)
    regime_momentum[downtrend_low_vol] = ret_20d[downtrend_low_vol] / vol_20d_mom[downtrend_low_vol].replace(0, np.nan)
    
    # Normal volatility + sideways
    normal_sideways = (volatility_regime == 'normal') & (trend_regime == 'sideways')
    regime_momentum[normal_sideways] = (ret_5d[normal_sideways] + ret_20d[normal_sideways]) / 2
    
    # Mixed regimes - weighted average
    mixed_mask = ~(uptrend_high_vol | downtrend_high_vol | uptrend_low_vol | downtrend_low_vol | normal_sideways)
    regime_strength = np.abs(ret_20d[mixed_mask]) + (vol_20d[mixed_mask] / vol_60d[mixed_mask].replace(0, np.nan))
    short_term_weight = np.where(vol_20d[mixed_mask] > vol_60d[mixed_mask], 0.6, 0.3)
    regime_momentum[mixed_mask] = (short_term_weight * ret_5d[mixed_mask] + 
                                  (1 - short_term_weight) * ret_20d[mixed_mask]) * regime_strength
    
    # Efficiency-Volume Confirmation
    eff_5d_change = df['price_efficiency'] / df['price_efficiency'].shift(5) - 1
    eff_20d_change = df['price_efficiency'] / df['price_efficiency'].shift(20) - 1
    efficiency_velocity = eff_5d_change - eff_20d_change
    
    # Efficiency-volume correlation
    eff_vol_corr = pd.Series(0.0, index=df.index)
    for i in range(20, len(df)):
        if i >= 20:
            window_eff = df['price_efficiency'].iloc[i-19:i+1]
            window_vol = df['volume'].iloc[i-19:i+1]
            if len(window_eff) == 20 and len(window_vol) == 20:
                eff_vol_corr.iloc[i] = window_eff.corr(window_vol)
    
    # Volume-regime efficiency weighting
    volume_weight = pd.Series(1.0, index=df.index)
    volume_weight[volume_regime == 'high'] = 1.3
    volume_weight[volume_regime == 'low'] = 0.7
    
    efficiency_factor = efficiency_velocity * eff_vol_corr * volume_weight
    
    # Range-based signals
    range_5d_mom = df['daily_range'] / df['daily_range'].shift(5) - 1
    range_20d_mom = df['daily_range'] / df['daily_range'].shift(20) - 1
    
    # Range-amount correlation
    range_amount_corr = pd.Series(0.0, index=df.index)
    for i in range(20, len(df)):
        if i >= 20:
            window_range = df['daily_range'].iloc[i-19:i+1]
            window_amount = df['amount'].iloc[i-19:i+1]
            if len(window_range) == 20 and len(window_amount) == 20:
                range_amount_corr.iloc[i] = window_range.corr(window_amount)
    
    amount_5d_mom = df['amount'] / df['amount'].shift(5) - 1
    amount_20d_mom = df['amount'] / df['amount'].shift(20) - 1
    
    range_amount_div_5d = range_5d_mom - amount_5d_mom
    range_amount_div_20d = range_20d_mom - amount_20d_mom
    
    alignment_score = range_amount_corr * (1 - np.abs(range_amount_div_5d + range_amount_div_20d) / 2)
    
    # Regime-contextual range signals
    range_signals = pd.Series(0.0, index=df.index)
    
    # High volatility range expansion
    high_vol_range = (volatility_regime == 'high') & (range_5d_mom > 0) & (range_amount_corr > 0.3)
    range_signals[high_vol_range] = range_5d_mom[high_vol_range] * range_amount_corr[high_vol_range]
    
    # Low volatility range contraction
    low_vol_range = (volatility_regime == 'low') & (range_5d_mom < 0) & (range_amount_corr > 0.2)
    range_signals[low_vol_range] = range_5d_mom[low_vol_range] * (1 + range_amount_corr[low_vol_range])
    
    # Normal volatility - mixed signals
    normal_vol_mask = volatility_regime == 'normal'
    range_signals[normal_vol_mask] = alignment_score[normal_vol_mask]
    
    # Multi-timeframe integration with regime-optimized weighting
    # Timeframe weights based on volatility regime
    short_term_weight = pd.Series(0.5, index=df.index)
    short_term_weight[volatility_regime == 'high'] = 0.6
    short_term_weight[volatility_regime == 'low'] = 0.3
    
    # Component weights based on trend regime
    momentum_weight = pd.Series(0.5, index=df.index)
    momentum_weight[trend_regime.isin(['uptrend', 'downtrend'])] = 0.7
    momentum_weight[trend_regime == 'sideways'] = 0.3
    
    efficiency_weight = 1 - momentum_weight
    
    # Short-term components (5-day)
    short_term_momentum = ret_5d / vol_5d.replace(0, np.nan)
    short_term_efficiency = eff_5d_change
    short_term_range = range_amount_div_5d
    
    # Medium-term components (20-day)
    medium_term_momentum = ret_20d / vol_20d_mom.replace(0, np.nan)
    medium_term_efficiency = eff_20d_change
    medium_term_range = range_amount_div_20d
    
    # Timeframe alignment scoring
    timeframe_alignment = (
        (np.sign(short_term_momentum) == np.sign(medium_term_momentum)) * 0.3 +
        (np.sign(short_term_efficiency) == np.sign(medium_term_efficiency)) * 0.3 +
        (np.sign(short_term_range) == np.sign(medium_term_range)) * 0.4
    )
    
    # Composite alpha construction
    short_term_composite = (
        short_term_momentum * momentum_weight +
        short_term_efficiency * efficiency_weight +
        short_term_range * 0.3
    )
    
    medium_term_composite = (
        medium_term_momentum * momentum_weight +
        medium_term_efficiency * efficiency_weight +
        medium_term_range * 0.3
    )
    
    # Final weighted combination
    alpha = (
        short_term_composite * short_term_weight +
        medium_term_composite * (1 - short_term_weight)
    )
    
    # Apply alignment enhancement
    alignment_multiplier = 1 + timeframe_alignment * 0.5
    alpha = alpha * alignment_multiplier
    
    # Apply regime clarity amplification
    regime_clarity = (
        (volatility_regime != 'normal').astype(int) * 0.3 +
        (trend_regime != 'sideways').astype(int) * 0.3 +
        (volume_regime != 'normal').astype(int) * 0.2 +
        (timeframe_alignment > 0.6).astype(int) * 0.2
    )
    
    clarity_multiplier = 1 + regime_clarity * 0.4
    alpha = alpha * clarity_multiplier
    
    # Normalize and clean
    alpha = alpha.replace([np.inf, -np.inf], np.nan)
    alpha = (alpha - alpha.rolling(60).mean()) / alpha.rolling(60).std()
    
    return alpha
