import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Multi-Timeframe Price-Volume Divergence Factor
    """
    # Volatility Estimation
    # Price Volatility
    close_ret = df['close'].pct_change()
    vol_5d_close = close_ret.rolling(window=5).std()
    vol_hl_range = (df['high'] - df['low']).rolling(window=10).mean()
    
    # Volume Volatility
    volume_ret = df['volume'].pct_change()
    vol_5d_volume = volume_ret.rolling(window=5).std()
    vol_10d_volume = df['volume'].rolling(window=10).std()
    
    # Multi-Timeframe Momentum Calculation
    # Short-term (3-day)
    mom_price_3d = df['close'] / df['close'].shift(3) - 1
    mom_volume_3d = df['volume'] / df['volume'].shift(3) - 1
    mom_amount_3d = df['amount'] / df['amount'].shift(3) - 1
    
    # Medium-term (10-day)
    mom_price_10d = df['close'] / df['close'].shift(10) - 1
    mom_volume_10d = df['volume'] / df['volume'].shift(10) - 1
    mom_amount_10d = df['amount'] / df['amount'].shift(10) - 1
    
    # Long-term (20-day)
    mom_price_20d = df['close'] / df['close'].shift(20) - 1
    mom_volume_20d = df['volume'] / df['volume'].shift(20) - 1
    mom_amount_20d = df['amount'] / df['amount'].shift(20) - 1
    
    # Volatility-Normalized Divergence
    # Price-Volume Divergence
    pv_div_short = (mom_price_3d - mom_volume_3d) / vol_5d_close.replace(0, np.nan)
    pv_div_medium = (mom_price_10d - mom_volume_10d) / vol_hl_range.replace(0, np.nan)
    pv_div_long = (mom_price_20d - mom_volume_20d) / vol_5d_close.rolling(window=20).mean().replace(0, np.nan)
    
    # Price-Amount Divergence
    pa_div_short = (mom_price_3d - mom_amount_3d) / vol_5d_close.replace(0, np.nan)
    pa_div_medium = (mom_price_10d - mom_amount_10d) / vol_hl_range.replace(0, np.nan)
    pa_div_long = (mom_price_20d - mom_amount_20d) / vol_5d_close.rolling(window=20).mean().replace(0, np.nan)
    
    # Volume-Amount Divergence
    va_div_short = (mom_volume_3d - mom_amount_3d) / vol_5d_volume.replace(0, np.nan)
    va_div_medium = (mom_volume_10d - mom_amount_10d) / vol_10d_volume.replace(0, np.nan)
    va_div_long = (mom_volume_20d - mom_amount_20d) / vol_5d_volume.rolling(window=20).mean().replace(0, np.nan)
    
    # Regime Detection
    # Volatility Regime
    vol_20d_percentile = vol_5d_close.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    high_vol_regime = vol_20d_percentile > 0.8
    low_vol_regime = vol_20d_percentile < 0.2
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Correlation Regime
    price_volume_corr = df['close'].rolling(window=10).corr(df['volume'])
    high_corr_regime = price_volume_corr > 0.2
    low_corr_regime = price_volume_corr < -0.2
    neutral_corr_regime = ~(high_corr_regime | low_corr_regime)
    
    # Trend Regime
    price_trend = np.sign(mom_price_10d)
    volume_trend = np.sign(mom_volume_10d)
    bullish_regime = (price_trend > 0) & (volume_trend > 0)
    bearish_regime = (price_trend < 0) & (volume_trend < 0)
    divergent_regime = ~(bullish_regime | bearish_regime)
    
    # Dynamic Factor Combination
    # Initialize weights
    timeframe_weights = pd.DataFrame(index=df.index, columns=['short', 'medium', 'long'])
    divergence_weights = pd.DataFrame(index=df.index, columns=['pv', 'pa', 'va'])
    
    # Volatility regime weighting
    timeframe_weights.loc[high_vol_regime, ['short', 'medium', 'long']] = [0.6, 0.3, 0.1]
    timeframe_weights.loc[low_vol_regime, ['short', 'medium', 'long']] = [0.1, 0.3, 0.6]
    timeframe_weights.loc[normal_vol_regime, ['short', 'medium', 'long']] = [0.4, 0.4, 0.2]
    
    # Correlation regime weighting
    divergence_weights.loc[high_corr_regime, ['pv', 'pa', 'va']] = [0.2, 0.5, 0.3]
    divergence_weights.loc[low_corr_regime, ['pv', 'pa', 'va']] = [0.5, 0.2, 0.3]
    divergence_weights.loc[neutral_corr_regime, ['pv', 'pa', 'va']] = [0.33, 0.33, 0.34]
    
    # Calculate weighted divergence scores
    pv_weighted = (pv_div_short * timeframe_weights['short'] + 
                   pv_div_medium * timeframe_weights['medium'] + 
                   pv_div_long * timeframe_weights['long']) * divergence_weights['pv']
    
    pa_weighted = (pa_div_short * timeframe_weights['short'] + 
                   pa_div_medium * timeframe_weights['medium'] + 
                   pa_div_long * timeframe_weights['long']) * divergence_weights['pa']
    
    va_weighted = (va_div_short * timeframe_weights['short'] + 
                   va_div_medium * timeframe_weights['medium'] + 
                   va_div_long * timeframe_weights['long']) * divergence_weights['va']
    
    # Final alpha factor
    alpha_factor = pv_weighted + pa_weighted + va_weighted
    
    # Trend regime adjustment
    alpha_factor = alpha_factor * np.where(bullish_regime, 1.2, 
                                          np.where(bearish_regime, 0.8, 1.0))
    
    # Volume confirmation
    volume_confirmation = np.where(mom_volume_10d > 0, 1.1, 0.9)
    alpha_factor = alpha_factor * volume_confirmation
    
    return alpha_factor
