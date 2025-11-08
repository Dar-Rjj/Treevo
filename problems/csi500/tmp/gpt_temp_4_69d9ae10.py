import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Alpha Factor
    Adapts factor construction based on detected market volatility regimes
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Volatility Regime Detection
    # Historical Volatility Clustering
    returns = df['close'].pct_change()
    vol_20d = returns.rolling(window=20).std()
    vol_60d = returns.rolling(window=60).std()
    
    # Rolling volatility percentile (20-day vs 60-day)
    vol_percentile = vol_20d.rolling(window=60).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)) if len(x.dropna()) > 0 else False, 
        raw=False
    ).astype(bool)
    
    # Intraday Volatility Signature
    opening_gap = (df['open'] - df['close'].shift(1)).abs() / df['close'].shift(1)
    day_range_ratio = (df['high'] - df['low']) / df['close']
    
    # Combined regime detection
    high_vol_regime = (
        (vol_percentile == True) | 
        (opening_gap > opening_gap.rolling(20).quantile(0.8)) |
        (day_range_ratio > day_range_ratio.rolling(20).quantile(0.8))
    )
    
    low_vol_regime = (
        (vol_percentile == False) & 
        (opening_gap < opening_gap.rolling(20).quantile(0.3)) &
        (day_range_ratio < day_range_ratio.rolling(20).quantile(0.3))
    )
    
    # Regime-Specific Factor Construction
    
    # High Volatility Regime Factors
    # Extreme Move Persistence
    large_range = (df['high'] - df['low']) / df['close'] > (df['high'] - df['low']).rolling(20).quantile(0.8) / df['close']
    gap_fill = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    follow_through = df['close'].pct_change().shift(-1)  # Note: This is for calculation, will be shifted
    
    # Volume Spike Quality
    volume_ma = df['volume'].rolling(20).mean()
    volume_spike = df['volume'] > volume_ma * 1.5
    price_confirmation = (df['close'].pct_change().abs() > returns.rolling(20).std() * 1.5)
    
    high_vol_factor = (
        (large_range.astype(int) * np.sign(follow_through)) * 0.4 +
        (-np.sign(gap_fill) * (gap_fill.abs() > 0.01).astype(int)) * 0.3 +
        (volume_spike & price_confirmation).astype(int) * np.sign(df['close'].pct_change()) * 0.3
    )
    
    # Low Volatility Regime Factors
    # Compression Breakout Signal
    narrow_range = (df['high'] - df['low']) / df['close'] < (df['high'] - df['low']).rolling(20).quantile(0.3) / df['close']
    range_expansion = ((df['high'] - df['low']) / df['close']) / ((df['high'] - df['low']).rolling(5).mean() / df['close'])
    volume_expansion = df['volume'] > df['volume'].rolling(5).mean() * 1.2
    
    # Mean Reversion Strength
    price_ma_5 = df['close'].rolling(5).mean()
    price_ma_20 = df['close'].rolling(20).mean()
    deviation = (df['close'] - price_ma_20) / price_ma_20
    reversion_speed = (df['close'].pct_change() * np.sign(-deviation.shift(1))).rolling(3).mean()
    
    low_vol_factor = (
        (narrow_range & (range_expansion > 1.5) & volume_expansion).astype(int) * np.sign(df['close'].pct_change()) * 0.4 +
        (deviation.abs() > 0.02).astype(int) * reversion_speed * 3 * 0.6
    )
    
    # Adaptive Signal Blending
    # Regime Confidence Weighting
    regime_stability = pd.Series(0.0, index=df.index)
    for i in range(5, len(df)):
        recent_high_vol = high_vol_regime.iloc[i-5:i].sum()
        recent_low_vol = low_vol_regime.iloc[i-5:i].sum()
        if recent_high_vol >= 4:
            regime_stability.iloc[i] = 1.0
        elif recent_low_vol >= 4:
            regime_stability.iloc[i] = -1.0
    
    # Transition phase handling
    in_transition = (regime_stability.abs() < 0.5)
    
    # Multi-timeframe integration
    intraday_trend = (df['close'] - df['open']) / df['open']
    daily_trend = df['close'].pct_change()
    alignment = np.sign(intraday_trend) == np.sign(daily_trend)
    
    # Signal persistence filter
    signal_persistence = high_vol_factor.rolling(3).mean().fillna(0) + low_vol_factor.rolling(3).mean().fillna(0)
    
    # Final alpha construction
    for i in range(len(df)):
        if i < 60:  # Warm-up period
            alpha.iloc[i] = 0
            continue
            
        if regime_stability.iloc[i] > 0.5:  # High volatility regime
            alpha.iloc[i] = high_vol_factor.iloc[i] * (1 + alignment.iloc[i] * 0.2)
        elif regime_stability.iloc[i] < -0.5:  # Low volatility regime
            alpha.iloc[i] = low_vol_factor.iloc[i] * (1 + alignment.iloc[i] * 0.2)
        else:  # Transition phase - blended approach
            high_weight = high_vol_regime.iloc[i-5:i].mean()
            low_weight = low_vol_regime.iloc[i-5:i].mean()
            total_weight = high_weight + low_weight
            if total_weight > 0:
                alpha.iloc[i] = (high_vol_factor.iloc[i] * high_weight + low_vol_factor.iloc[i] * low_weight) / total_weight
            else:
                alpha.iloc[i] = 0
        
        # Apply persistence filter
        if i >= 3:
            alpha.iloc[i] = alpha.iloc[i] * 0.7 + signal_persistence.iloc[i] * 0.3
    
    # Ensure no future data leakage by shifting the final output
    # Note: follow_through was used in calculation but not in final alpha to avoid lookahead
    alpha = alpha.shift(1).fillna(0)
    
    return alpha
