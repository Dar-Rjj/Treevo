import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate mid-price
    df['mid_price'] = (df['high'] + df['low']) / 2
    
    # Bidirectional Volatility Analysis
    df['mid_ret'] = df['mid_price'].pct_change()
    df['pos_ret'] = df['mid_ret'].where(df['mid_ret'] > 0, 0)
    df['neg_ret'] = df['mid_ret'].where(df['mid_ret'] < 0, 0)
    
    upside_vol = df['pos_ret'].rolling(window=30).std()
    downside_vol = df['neg_ret'].rolling(window=30).std()
    
    # Multi-Timeframe Range Volatility
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    range_vol_5 = df['daily_range'].rolling(window=5).mean()
    return_vol_5 = df['mid_ret'].rolling(window=5).std()
    
    range_vol_10 = df['daily_range'].rolling(window=10).mean()
    return_vol_10 = df['mid_ret'].rolling(window=10).std()
    
    range_vol_20 = df['daily_range'].rolling(window=20).mean()
    return_vol_20 = df['mid_ret'].rolling(window=20).std()
    
    # Volatility Regime Classification
    vol_asymmetry = upside_vol - downside_vol
    bull_regime = (vol_asymmetry > 0).astype(int)
    bear_regime = (vol_asymmetry < 0).astype(int)
    neutral_regime = (vol_asymmetry == 0).astype(int)
    
    # Volatility level classification
    vol_level_20 = df['mid_ret'].rolling(window=20).std()
    vol_quantile = vol_level_20.rolling(window=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    high_vol = (vol_quantile > 0.7).astype(int)
    low_vol = (vol_quantile < 0.3).astype(int)
    normal_vol = ((vol_quantile >= 0.3) & (vol_quantile <= 0.7)).astype(int)
    
    # Efficiency Divergence Calculation
    def calculate_divergence(price_window, vol_window):
        # Price momentum
        price_ret = df['mid_price'].pct_change(periods=price_window)
        vol_adjusted_ret = price_ret / df['mid_ret'].rolling(window=vol_window).std()
        
        # Volume flow
        df['price_range'] = df['high'] - df['low']
        df['price_range'] = df['price_range'].replace(0, np.nan)
        volume_flow = ((df['close'] - df['open']) / df['price_range']) * df['volume']
        volume_momentum = volume_flow.rolling(window=price_window).mean()
        
        # Divergence
        divergence = np.sign(volume_momentum - vol_adjusted_ret)
        return divergence, vol_adjusted_ret
    
    # Short-term divergences
    div_3, mom_3 = calculate_divergence(3, 5)
    div_5, mom_5 = calculate_divergence(5, 5)
    
    # Medium-term divergences
    div_8, mom_8 = calculate_divergence(8, 10)
    div_15, mom_15 = calculate_divergence(15, 10)
    
    # Long-term divergences
    div_20, mom_20 = calculate_divergence(20, 20)
    
    # Cross-Timeframe Confirmation
    short_consistency = (div_3 == div_5).astype(int)
    medium_consistency = (div_8 == div_15).astype(int)
    multi_timeframe_alignment = ((div_3 == div_8) & (div_8 == div_20)).astype(int)
    
    # Regime-Adaptive Weighting
    # Volatility asymmetry weighting
    regime_divergence = (
        bull_regime * np.maximum(0, div_3 + div_5 + div_8 + div_15 + div_20) +
        bear_regime * np.minimum(0, div_3 + div_5 + div_8 + div_15 + div_20) +
        neutral_regime * (div_3 + div_5 + div_8 + div_15 + div_20)
    )
    
    # Volatility level weighting
    timeframe_weights = (
        high_vol * (0.4 * div_3 + 0.3 * div_5 + 0.2 * div_8 + 0.1 * div_15) +
        low_vol * (0.1 * div_3 + 0.2 * div_5 + 0.3 * div_8 + 0.4 * div_15) +
        normal_vol * (0.25 * div_3 + 0.25 * div_5 + 0.25 * div_8 + 0.25 * div_15)
    )
    
    regime_weighted_divergence = regime_divergence * timeframe_weights
    
    # Momentum strength
    momentum_strength = (
        np.abs(mom_3.fillna(0)) + 
        np.abs(mom_5.fillna(0)) + 
        np.abs(mom_8.fillna(0)) + 
        np.abs(mom_15.fillna(0)) + 
        np.abs(mom_20.fillna(0))
    ) / 5
    
    # Liquidity pressure
    range_per_volume = (df['high'] - df['low']) / df['volume']
    liquidity_pressure = range_per_volume / range_per_volume.rolling(window=60).median()
    
    # Composite Alpha Generation
    confirmation_factor = (short_consistency + medium_consistency + multi_timeframe_alignment) / 3
    
    final_factor = (
        regime_weighted_divergence * 
        momentum_strength * 
        (1 / liquidity_pressure) * 
        confirmation_factor
    )
    
    return final_factor
