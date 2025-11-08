import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-adaptive alpha factor combining momentum, order flow, breakout efficiency, 
    and mean reversion signals with dynamic weighting based on volatility regimes.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price features
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    amount = df['amount']
    volume = df['volume']
    
    # Volatility Regime Detection
    def calculate_volatility_regime():
        # Short-term volatility measures
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - close.shift(1)), 
                                 abs(low - close.shift(1))))
        tr_3d_avg = tr.rolling(window=3).mean()
        vol_ratio_1 = tr_3d_avg / close.shift(3)
        
        ret_5d_std = close.pct_change().rolling(window=5).std()
        
        # 20-day volatility percentiles
        vol_combined = (vol_ratio_1 + ret_5d_std) / 2
        vol_20d_20pct = vol_combined.rolling(window=20).quantile(0.2)
        vol_20d_80pct = vol_combined.rolling(window=20).quantile(0.8)
        
        # Regime classification
        high_vol = vol_combined > vol_20d_80pct
        low_vol = vol_combined < vol_20d_20pct
        normal_vol = ~high_vol & ~low_vol
        
        # Regime persistence
        regime_persistence = pd.Series(0, index=df.index)
        current_regime = None
        persistence_count = 0
        
        for i in range(len(df)):
            if high_vol.iloc[i]:
                regime = 'high'
            elif low_vol.iloc[i]:
                regime = 'low'
            else:
                regime = 'normal'
            
            if regime == current_regime:
                persistence_count += 1
            else:
                persistence_count = 1
                current_regime = regime
            
            regime_persistence.iloc[i] = persistence_count
        
        regime_stability = regime_persistence.rolling(window=5).mean()
        
        return high_vol, low_vol, normal_vol, regime_stability
    
    # Multi-Timeframe Momentum Quality
    def calculate_momentum_quality():
        # Intraday momentum quality (1-day)
        price_efficiency = abs(close - open_price) / (high - low).replace(0, np.nan)
        volume_confirmation = amount / (abs(close - open_price).replace(0, np.nan) * volume.replace(0, np.nan))
        gap_quality = (open_price - close.shift(1)) / close.shift(1)
        
        intraday_score = (price_efficiency.rank(pct=True) + 
                         volume_confirmation.rank(pct=True) + 
                         gap_quality.rank(pct=True)) / 3
        
        # Short-term momentum quality (3-5 day)
        ret_3d = close.pct_change(3)
        ret_5d = close.pct_change(5)
        
        # Return consistency
        sign_consistency = ((ret_3d > 0) & (ret_5d > 0)) | ((ret_3d < 0) & (ret_5d < 0))
        
        # Volume-price alignment
        vol_5d = volume.rolling(window=5).mean()
        price_5d = close.rolling(window=5).mean()
        volume_price_corr = volume.rolling(window=5).corr(close)
        
        # Range efficiency
        cumulative_tr = (high - low).rolling(window=5).sum()
        range_efficiency = ret_5d / cumulative_tr.replace(0, np.nan)
        
        short_term_score = (sign_consistency.astype(float) + 
                          volume_price_corr.rank(pct=True) + 
                          range_efficiency.rank(pct=True)) / 3
        
        # Medium-term momentum quality (10-15 day)
        ret_10d = close.pct_change(10)
        ret_15d = close.pct_change(15)
        
        # Trend smoothness
        ret_var = close.pct_change().rolling(window=10).var()
        trend_smoothness = abs(ret_10d) / (ret_var.replace(0, np.nan) + 1e-8)
        
        # Volume distribution
        vol_peaks = (volume == volume.rolling(window=15, center=True).max())
        price_highs = (close == close.rolling(window=15, center=True).max())
        volume_distribution = (vol_peaks & price_highs).astype(float)
        
        # Acceleration profile
        mom_acceleration = ret_10d - ret_5d.shift(5)
        
        medium_term_score = (trend_smoothness.rank(pct=True) + 
                           volume_distribution.rank(pct=True) + 
                           mom_acceleration.rank(pct=True)) / 3
        
        return intraday_score, short_term_score, medium_term_score
    
    # Quality-Weighted Order Flow Framework
    def calculate_order_flow_quality():
        # Flow direction analysis
        up_days = close > close.shift(1)
        down_days = close < close.shift(1)
        
        up_amount_ratio = (amount * up_days).rolling(window=5).sum() / amount.rolling(window=5).sum()
        down_amount_ratio = (amount * down_days).rolling(window=5).sum() / amount.rolling(window=5).sum()
        net_flow_score = up_amount_ratio - down_amount_ratio
        
        # Flow persistence
        flow_direction = (amount * (close.diff() > 0)).rolling(window=3).sum() - \
                        (amount * (close.diff() < 0)).rolling(window=3).sum()
        flow_persistence = (flow_direction > 0).astype(int).rolling(window=5).sum()
        
        # Flow concentration
        amount_per_volume = amount / volume.replace(0, np.nan)
        flow_concentration = amount_per_volume.rolling(window=5).std() / amount_per_volume.rolling(window=5).mean()
        
        # Price impact efficiency
        price_impact = amount / abs(close.diff()).replace(0, np.nan)
        flow_efficiency = price_impact.rolling(window=3).mean()
        
        # Timing quality
        price_turning = ((close > close.shift(1)) & (close.shift(1) < close.shift(2))) | \
                       ((close < close.shift(1)) & (close.shift(1) > close.shift(2)))
        flow_timing = (amount * price_turning).rolling(window=5).sum() / amount.rolling(window=5).sum()
        
        # Consistency metrics
        directional_consistency = (close.diff() > 0).rolling(window=5).mean() - 0.5
        
        flow_quality = (net_flow_score.rank(pct=True) + 
                       flow_persistence.rank(pct=True) + 
                       (1 - flow_concentration).rank(pct=True) + 
                       flow_efficiency.rank(pct=True) + 
                       flow_timing.rank(pct=True) + 
                       directional_consistency.rank(pct=True)) / 6
        
        return flow_quality
    
    # Multi-Timeframe Breakout Efficiency
    def calculate_breakout_efficiency():
        # Short-term breakout (1-3 day)
        range_expansion = (high - low) / (high - low).rolling(window=5).mean()
        volume_surge = volume / volume.rolling(window=5).mean()
        gap_breakout = abs(open_price - close.shift(1)) / close.shift(1)
        
        short_term_breakout = (range_expansion.rank(pct=True) + 
                             volume_surge.rank(pct=True) + 
                             gap_breakout.rank(pct=True)) / 3
        
        # Medium-term breakout (5-10 day)
        key_level_violation = (close > close.rolling(window=10).max()) | (close < close.rolling(window=10).min())
        trend_break = abs(close - close.rolling(window=10).mean()) / close.rolling(window=10).std()
        
        medium_term_breakout = (key_level_violation.astype(float).rank(pct=True) + 
                              trend_break.rank(pct=True)) / 2
        
        # Efficiency metrics
        breakout_follow_through = (close.shift(-1) - close) / (high - low).replace(0, np.nan)
        volume_efficiency = volume / abs(close.diff()).replace(0, np.nan)
        
        breakout_efficiency = (short_term_breakout + 
                             medium_term_breakout + 
                             breakout_follow_through.rank(pct=True) + 
                             volume_efficiency.rank(pct=True)) / 4
        
        return breakout_efficiency
    
    # Adaptive Mean Reversion System
    def calculate_mean_reversion():
        # Multi-timeframe overextension
        # Short-term extremes
        intraday_deviation = (close - (high + low) / 2) / ((high - low) / 2).replace(0, np.nan)
        mom_2d_extreme = close.pct_change(2).rank(pct=True)
        volume_divergence = (volume / volume.rolling(window=5).mean()) - (abs(close.pct_change()) / abs(close.pct_change()).rolling(window=5).mean())
        
        # Medium-term deviation
        price_vs_avg = (close - close.rolling(window=10).mean()) / close.rolling(window=10).std()
        cumulative_extreme = close.pct_change(10).rank(pct=True)
        
        overextension_score = (intraday_deviation.rank(pct=True) + 
                             mom_2d_extreme + 
                             volume_divergence.rank(pct=True) + 
                             price_vs_avg.rank(pct=True) + 
                             cumulative_extreme) / 5
        
        # Mean reversion signal (negative for overbought, positive for oversold)
        mean_reversion = -overextension_score
        
        return mean_reversion
    
    # Calculate all components
    high_vol, low_vol, normal_vol, regime_stability = calculate_volatility_regime()
    intraday_mom, short_term_mom, medium_term_mom = calculate_momentum_quality()
    flow_quality = calculate_order_flow_quality()
    breakout_eff = calculate_breakout_efficiency()
    mean_rev = calculate_mean_reversion()
    
    # Dynamic factor integration with regime-adaptive weighting
    for i in range(len(df)):
        if i < 20:  # Minimum data requirement
            result.iloc[i] = 0
            continue
            
        # Regime-optimized combination
        if high_vol.iloc[i]:
            momentum_weight = 0.7 * intraday_mom.iloc[i] + 0.2 * short_term_mom.iloc[i] + 0.1 * medium_term_mom.iloc[i]
            flow_weight = 0.6
            breakout_weight = 0.3
            reversion_weight = 0.4
        elif low_vol.iloc[i]:
            momentum_weight = 0.1 * intraday_mom.iloc[i] + 0.3 * short_term_mom.iloc[i] + 0.6 * medium_term_mom.iloc[i]
            flow_weight = 0.3
            breakout_weight = 0.6
            reversion_weight = 0.5
        else:  # normal volatility
            momentum_weight = 0.4 * intraday_mom.iloc[i] + 0.4 * short_term_mom.iloc[i] + 0.2 * medium_term_mom.iloc[i]
            flow_weight = 0.5
            breakout_weight = 0.4
            reversion_weight = 0.3
        
        # Apply regime stability adjustment
        stability_adj = min(regime_stability.iloc[i] / 5, 1.0)
        
        # Combine factors with regime-adaptive weights
        combined_factor = (
            momentum_weight * stability_adj +
            flow_weight * flow_quality.iloc[i] * stability_adj +
            breakout_weight * breakout_eff.iloc[i] * stability_adj +
            reversion_weight * mean_rev.iloc[i] * (1 - stability_adj)
        )
        
        result.iloc[i] = combined_factor
    
    # Normalize the final factor
    result = (result - result.rolling(window=20).mean()) / result.rolling(window=20).std()
    
    return result.fillna(0)
