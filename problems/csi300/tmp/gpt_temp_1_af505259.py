import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Price-Volume Divergence Factor
    Combines volatility regime analysis with price-volume relationship dynamics
    to generate regime-adaptive trading signals.
    """
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Calculation
    vol_10d = returns.rolling(window=10).std()
    vol_momentum = vol_10d.pct_change(3)  # 3-day change in volatility
    
    # Volatility regime classification
    vol_regime = pd.Series(np.where(vol_10d > vol_10d.rolling(window=20).quantile(0.7), 
                                   'high', 
                                   np.where(vol_10d < vol_10d.rolling(window=20).quantile(0.3), 
                                           'low', 'normal')), 
                          index=df.index)
    
    # Price-Volume Relationship Analysis
    # 10-day rolling price-volume correlation
    price_vol_corr = df['close'].rolling(window=10).corr(df['volume'])
    
    # Volume-weighted price deviation
    price_ma_5d = df['close'].rolling(window=5).mean()
    price_deviation = (df['close'] - price_ma_5d) / price_ma_5d
    volume_weighted_deviation = (price_deviation * df['volume']).rolling(window=5).sum() / df['volume'].rolling(window=5).sum()
    
    # Divergence patterns
    price_trend_5d = df['close'].pct_change(5)
    volume_trend_5d = df['volume'].pct_change(5)
    
    # Divergence signals
    weak_trend = ((price_trend_5d > 0) & (volume_trend_5d < 0)).astype(int)
    strong_downtrend = ((price_trend_5d < 0) & (volume_trend_5d > 0)).astype(int)
    
    # Construct Divergence Signals
    # Volatility-expansion divergence
    high_vol_divergence = ((vol_regime == 'high') & 
                          (price_vol_corr < -0.3) & 
                          (np.abs(price_deviation) > 0.02)).astype(int)
    
    low_vol_convergence = ((vol_regime == 'low') & 
                          (price_vol_corr > 0.5) & 
                          (np.abs(price_deviation) < 0.01)).astype(int)
    
    # Multi-timeframe divergence confirmation
    price_vol_corr_5d = df['close'].rolling(window=5).corr(df['volume'])
    multi_timeframe_divergence = ((price_vol_corr < -0.2) & 
                                (price_vol_corr_5d < -0.2) & 
                                (vol_momentum > 0)).astype(int)
    
    # Volume-intensity adjusted signals
    volume_quantile = df['volume'].rolling(window=20).apply(lambda x: pd.Series(x).quantile(0.7), raw=True)
    high_volume_days = (df['volume'] > volume_quantile).astype(int)
    
    volume_intensity_signals = ((high_volume_days == 1) & 
                              (np.abs(volume_weighted_deviation) > 0.015) & 
                              (price_vol_corr < 0)).astype(int)
    
    # Generate Regime-Adaptive Factors
    # High volatility regime factors
    high_vol_factors = (
        (vol_regime == 'high').astype(int) * 
        (-volume_weighted_deviation * vol_momentum * 2 +  # Mean reversion after spikes
         high_vol_divergence * -0.5 +  # Reversal signals
         volume_intensity_signals * 0.3)  # Volume confirmation
    )
    
    # Low volatility regime factors  
    low_vol_factors = (
        (vol_regime == 'low').astype(int) * 
        (volume_weighted_deviation * 1.5 +  # Breakout anticipation
         low_vol_convergence * 0.4 +  # Compression signals
         (price_vol_corr > 0.6).astype(int) * 0.2)  # Trend strength
    )
    
    # Normal regime factors
    normal_vol_factors = (
        (vol_regime == 'normal').astype(int) * 
        (volume_weighted_deviation * 0.8 +
         multi_timeframe_divergence * -0.3 +
         weak_trend * -0.2 +
         strong_downtrend * -0.4)
    )
    
    # Combine all factors with regime weighting
    combined_factor = (
        high_vol_factors.fillna(0) + 
        low_vol_factors.fillna(0) + 
        normal_vol_factors.fillna(0)
    )
    
    # Smooth the factor with 3-day moving average
    final_factor = combined_factor.rolling(window=3).mean()
    
    return final_factor
