import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate mid-price
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # 1. Volatility Regime Classification with Asymmetry Analysis
    # Calculate returns
    data['mid_return'] = data['mid_price'].pct_change()
    
    # Bidirectional Volatility (30-day window)
    upside_vol = data['mid_return'].rolling(window=30).apply(lambda x: x[x > 0].std(), raw=False)
    downside_vol = data['mid_return'].rolling(window=30).apply(lambda x: x[x < 0].std(), raw=False)
    
    # Volatility Asymmetry
    vol_asymmetry = (upside_vol / downside_vol) - 1
    
    # Multi-Timeframe Range Volatility Classification
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    
    # Short-term (5-day)
    range_vol_5d = data['daily_range_vol'].rolling(window=5).mean()
    close_vol_5d = data['close'].pct_change().rolling(window=5).std()
    vol_regime_5d = np.where(range_vol_5d > close_vol_5d, 2, 
                            np.where(range_vol_5d < close_vol_5d, 0, 1))
    
    # Medium-term (10-day)
    range_vol_10d = data['daily_range_vol'].rolling(window=10).mean()
    close_vol_10d = data['close'].pct_change().rolling(window=10).std()
    vol_regime_10d = np.where(range_vol_10d > close_vol_10d, 2, 
                             np.where(range_vol_10d < close_vol_10d, 0, 1))
    
    # Long-term (20-day)
    range_vol_20d = data['daily_range_vol'].rolling(window=20).mean()
    close_vol_20d = data['close'].pct_change().rolling(window=20).std()
    vol_regime_20d = np.where(range_vol_20d > close_vol_20d, 2, 
                             np.where(range_vol_20d < close_vol_20d, 0, 1))
    
    # 2. Multi-Scale Efficiency Divergence with Synchronization
    # Directional Volume Flow
    data['directional_volume_flow'] = ((data['close'] - data['open']) / 
                                      (data['high'] - data['low'] + 1e-8)) * data['volume']
    
    # Short-Term Divergence (3-day, 5-day)
    # Volatility calculations
    vol_3d = data['mid_price'].pct_change().rolling(window=3).std()
    vol_5d = data['mid_price'].pct_change().rolling(window=5).std()
    
    # Price momentum
    price_momentum_3d = data['mid_price'].pct_change(periods=3) / (vol_3d + 1e-8)
    price_momentum_5d = data['mid_price'].pct_change(periods=5) / (vol_5d + 1e-8)
    
    # Volume flow momentum
    vol_flow_momentum_3d = data['directional_volume_flow'].rolling(window=3).mean()
    vol_flow_momentum_5d = data['directional_volume_flow'].rolling(window=5).mean()
    
    # Efficiency divergence
    eff_div_3d = np.sign(vol_flow_momentum_3d - price_momentum_3d)
    eff_div_5d = np.sign(vol_flow_momentum_5d - price_momentum_5d)
    sync_strength_3d = np.abs(price_momentum_3d) * np.abs(vol_flow_momentum_3d)
    sync_strength_5d = np.abs(price_momentum_5d) * np.abs(vol_flow_momentum_5d)
    
    # Medium-Term Divergence (8-day, 15-day)
    vol_8d = data['mid_price'].pct_change().rolling(window=8).std()
    vol_15d = data['mid_price'].pct_change().rolling(window=15).std()
    
    price_momentum_8d = data['mid_price'].pct_change(periods=8) / (vol_8d + 1e-8)
    price_momentum_15d = data['mid_price'].pct_change(periods=15) / (vol_15d + 1e-8)
    
    vol_flow_momentum_8d = data['directional_volume_flow'].rolling(window=8).mean()
    vol_flow_momentum_15d = data['directional_volume_flow'].rolling(window=15).mean()
    
    eff_div_8d = np.sign(vol_flow_momentum_8d - price_momentum_8d)
    eff_div_15d = np.sign(vol_flow_momentum_15d - price_momentum_15d)
    sync_strength_8d = np.abs(price_momentum_8d) * np.abs(vol_flow_momentum_8d)
    sync_strength_15d = np.abs(price_momentum_15d) * np.abs(vol_flow_momentum_15d)
    
    # Long-Term Divergence (20-day)
    vol_20d = data['mid_price'].pct_change().rolling(window=20).std()
    price_momentum_20d = data['mid_price'].pct_change(periods=20) / (vol_20d + 1e-8)
    vol_flow_momentum_20d = data['directional_volume_flow'].rolling(window=20).mean()
    eff_div_20d = np.sign(vol_flow_momentum_20d - price_momentum_20d)
    sync_strength_20d = np.abs(price_momentum_20d) * np.abs(vol_flow_momentum_20d)
    
    # 3. Cross-Timeframe Confirmation with Divergence Patterns
    # Short-term consistency
    short_term_match = (eff_div_3d == eff_div_5d).astype(int)
    short_term_confirmation = np.where(eff_div_3d == eff_div_5d, 
                                      np.where(eff_div_3d > 0, 1, -1), 0)
    
    # Medium-term consistency
    medium_term_match = (eff_div_8d == eff_div_15d).astype(int)
    medium_term_confirmation = np.where(eff_div_8d == eff_div_15d, 
                                       np.where(eff_div_8d > 0, 1, -1), 0)
    
    # Multi-timeframe alignment
    all_divergences = np.stack([eff_div_3d, eff_div_5d, eff_div_8d, eff_div_15d, eff_div_20d], axis=1)
    positive_count = (all_divergences > 0).sum(axis=1)
    negative_count = (all_divergences < 0).sum(axis=1)
    
    multi_timeframe_alignment = np.where(positive_count == 5, 2,
                                        np.where(negative_count == 5, -2,
                                                np.where(positive_count >= 3, 1,
                                                        np.where(negative_count >= 3, -1, 0))))
    
    # Divergence pattern detection
    bullish_div = ((price_momentum_20d < 0) & (vol_flow_momentum_20d > 0)).astype(int)
    bearish_div = ((price_momentum_20d > 0) & (vol_flow_momentum_20d < 0)).astype(int)
    
    # 4. Regime-Adaptive Signal Processing and Weighting
    # Volatility Asymmetry Regime
    bull_regime = (vol_asymmetry > 0.2).astype(int)
    bear_regime = (vol_asymmetry < -0.2).astype(int)
    neutral_regime = ((vol_asymmetry >= -0.2) & (vol_asymmetry <= 0.2)).astype(int)
    
    # Multi-Timeframe Volatility Regime Weighting
    high_vol_weight = ((vol_regime_5d == 2) | (vol_regime_10d == 2) | (vol_regime_20d == 2)).astype(int)
    low_vol_weight = ((vol_regime_5d == 0) | (vol_regime_10d == 0) | (vol_regime_20d == 0)).astype(int)
    normal_vol_weight = ((vol_regime_5d == 1) & (vol_regime_10d == 1) & (vol_regime_20d == 1)).astype(int)
    
    # Confirmation Strength Multiplier
    confirmation_strength = np.where(multi_timeframe_alignment == 2, 1.5,
                                    np.where(np.abs(multi_timeframe_alignment) == 1, 1.0,
                                            np.where((short_term_match == 1) | (medium_term_match == 1), 0.5, 0.2)))
    
    # 5. Composite Alpha Generation with Momentum Integration
    # Momentum Strength
    momentum_strength = (np.abs(price_momentum_3d) + np.abs(price_momentum_5d) + 
                        np.abs(price_momentum_8d) + np.abs(price_momentum_15d) + 
                        np.abs(price_momentum_20d)) / 5
    momentum_strength = np.cbrt(momentum_strength + 1e-8)
    
    # Liquidity-Adjusted Signal
    volume_weighted_price_impact = (data['high'] - data['low']) / (data['volume'] + 1e-8)
    relative_liquidity = volume_weighted_price_impact / volume_weighted_price_impact.rolling(window=60).median()
    liquidity_pressure = relative_liquidity - 1
    
    # Regime-Weighted Combination
    divergence_component = (eff_div_3d + eff_div_5d + eff_div_8d + eff_div_15d + eff_div_20d) / 5
    synchronization_component = (sync_strength_3d + sync_strength_5d + sync_strength_8d + 
                                sync_strength_15d + sync_strength_20d) / 5
    
    regime_weighted_combination = (
        bull_regime * (0.6 * divergence_component + 0.4 * synchronization_component) +
        bear_regime * (0.4 * divergence_component + 0.6 * synchronization_component) +
        neutral_regime * (0.5 * divergence_component + 0.5 * synchronization_component)
    )
    
    # Final Factor
    final_factor = (
        regime_weighted_combination * 
        momentum_strength * 
        liquidity_pressure * 
        confirmation_strength
    )
    
    return pd.Series(final_factor, index=data.index)
