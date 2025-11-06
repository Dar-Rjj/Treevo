import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper functions
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.abs(high - close_prev), np.abs(low - close_prev))
    
    def efficiency_ratio(close, window):
        price_change = close.diff(window).abs()
        volatility = close.diff().abs().rolling(window).sum()
        return price_change / (volatility + 1e-8)
    
    def fractal_dimension(high, low, close, window):
        range_avg = (high.rolling(window).max() - low.rolling(window).min())
        price_range = (high - low).rolling(window).sum()
        return np.log(price_range + 1e-8) / (np.log(range_avg + 1e-8) + 1e-8)
    
    # Calculate intermediate components
    close_prev = data['close'].shift(1)
    high_prev = data['high'].shift(1)
    low_prev = data['low'].shift(1)
    volume_prev = data['volume'].shift(1)
    
    # Fractal dimensions and efficiency ratios
    fd_3d = fractal_dimension(data['high'], data['low'], data['close'], 3)
    fd_5d = fractal_dimension(data['high'], data['low'], data['close'], 5)
    fd_8d = fractal_dimension(data['high'], data['low'], data['close'], 8)
    
    er_3d = efficiency_ratio(data['close'], 3)
    er_5d = efficiency_ratio(data['close'], 5)
    er_8d = efficiency_ratio(data['close'], 8)
    er_10d = efficiency_ratio(data['close'], 10)
    
    # 1. Multi-Timeframe Fractal Reversal
    short_term_fr = (data['close'].rolling(3).std() / (data['close'].rolling(3).mean() + 1e-8)) * fd_3d
    medium_term_fr = ((data['high'].rolling(5).max() - data['low'].rolling(5).min()) / (data['close'].rolling(5).mean() + 1e-8)) * er_8d
    fractal_reversal_div = np.abs(short_term_fr - medium_term_fr)
    
    # 2. Intraday Fractal Reversal Patterns
    opening_gap_rev = ((data['open'] - close_prev) / (data['high'] - data['low'] + 1e-8)) * fd_3d
    closing_eff_rev = (np.abs(data['close'] - (data['high'] + data['low'])/2) / (data['high'] - data['low'] + 1e-8)) * er_5d
    total_intraday_rev = opening_gap_rev + closing_eff_rev
    
    # 3. Fractal Momentum Divergence
    fractal_acc = (er_3d / (er_8d + 1e-8)) * fd_5d
    
    high_prox_rev = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * fd_3d
    low_prox_rev = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * fd_3d
    
    fractal_rev_intensity = np.where(
        (high_prox_rev < 0.3) & (low_prox_rev < 0.3), 
        1.5 * er_5d, 
        1.0
    )
    
    fractal_rev_momentum = fractal_acc * fractal_rev_intensity
    
    # 4. Volume-Fractal Regime Analysis
    vol_concentration_fractal = (data['volume'] / data['volume'].rolling(3).max()) * fd_3d
    
    vol_avg_5d = data['volume'].rolling(5).mean()
    vol_persistence = data['volume'].rolling(5).apply(
        lambda x: (x > vol_avg_5d.loc[x.index]).sum(), raw=False
    ) * er_5d
    
    fractal_volume_score = vol_concentration_fractal * vol_persistence
    
    # 5. Fractal Liquidity Dynamics
    fractal_spread_proxy = ((data['high'] - data['low']) / (data['close'] + 1e-8)) * er_3d
    fractal_trading_intensity = (data['volume'] / (data['high'] - data['low'] + 1e-8)) * fd_3d
    fractal_liquidity_cost = fractal_spread_proxy * fractal_trading_intensity
    
    # 6. Fractal Order Flow Characteristics
    fractal_price_impact = (np.abs(data['close'] - data['open']) / (data['volume'] + 1e-8)) * er_5d
    fractal_volume_clustering = (data['volume'] / data['volume'].rolling(3).max()) * fd_3d
    fractal_directional_intensity = ((data['close'] - data['open']) * data['volume'] / 
                                   (data['volume'].rolling(3).mean() + 1e-8)) * er_5d
    
    # 7. Volatility-Fractal Breakout Signals
    tr = true_range(data['high'], data['low'], close_prev)
    fractal_vol_ratio = (tr.rolling(10).std() / (tr.rolling(60).std() + 1e-8)) * fd_8d
    fractal_range_expansion = ((data['high'] - data['low']) / (high_prev - low_prev + 1e-8)) * er_5d
    
    close_std_3d = data['close'].rolling(3).std()
    close_std_6d = data['close'].rolling(6).std()
    fractal_vol_persistence = (close_std_3d / (close_std_6d + 1e-8)) * fd_5d
    
    fractal_range_breakout = (np.sign(data['close'] - close_prev) * 
                            ((data['high'] > high_prev) | (data['low'] < low_prev)).astype(float)) * er_5d
    fractal_gap_analysis = ((data['open'] - close_prev) / 
                           (np.abs(close_prev - data['close'].shift(5)) + 1e-8)) * fd_5d
    fractal_breakout_strength = fractal_range_breakout * fractal_gap_analysis
    
    fractal_range_regime = ((data['high'] - data['low']) / 
                           ((data['high'].rolling(5).max() - data['low'].rolling(5).min()).shift(1) + 1e-8)) * er_5d
    
    fractal_vol_alignment = np.where(
        fractal_vol_ratio > 1.0,
        fractal_range_expansion,
        1.0 / (fractal_range_expansion + 1e-8)
    )
    fractal_vol_score = fractal_vol_alignment * fractal_vol_persistence
    
    # 8. Microstructure-Fractal Integration
    intraday_fractal_eff = (np.abs(data['close'] - close_prev) / (data['high'] - data['low'] + 1e-8)) * fd_3d
    fractal_market_depth = (data['amount'] / (data['volume'] + 1e-8)) * er_5d
    
    prev_intraday_eff = (np.abs(close_prev - data['open'].shift(1)) / 
                        (high_prev - low_prev + 1e-8))
    fractal_eff_momentum = (intraday_fractal_eff / (prev_intraday_eff + 1e-8)) * fd_3d
    
    fractal_vol_momentum = (data['volume'] / (volume_prev + 1e-8) - 
                           (data['close'] / (close_prev + 1e-8) - 1)) * er_5d
    
    vol_ratio_t3 = data['volume'] / (data['volume'].shift(3) + 1e-8)
    vol_ratio_t4 = volume_prev / (data['volume'].shift(4) + 1e-8)
    fractal_liq_momentum = (vol_ratio_t3 - vol_ratio_t4) * fd_3d
    
    fractal_alignment_score = np.where(
        fractal_vol_momentum * fractal_liq_momentum > 0,
        1.2 * er_5d,
        0.8
    )
    
    fractal_intensity_trend = ((data['volume'] / (data['high'] - data['low'] + 1e-8)) / 
                              (volume_prev / (high_prev - low_prev + 1e-8))) * fd_3d
    fractal_eff_trend = (intraday_fractal_eff / (prev_intraday_eff + 1e-8)) * er_5d
    fractal_microstructure_score = fractal_intensity_trend * fractal_eff_trend * fractal_alignment_score
    
    # 9. Adaptive Multi-Fractal Reversal Alpha Construction
    core_fractal_reversal = fractal_reversal_div * total_intraday_rev
    volume_confirmed = core_fractal_reversal * fractal_volume_score * fractal_directional_intensity
    liquidity_adjusted = volume_confirmed / (fractal_liquidity_cost + 0.001)
    volatility_scaled = liquidity_adjusted * fractal_vol_score
    breakout_integrated = volatility_scaled * fractal_breakout_strength * fractal_range_regime
    microstructure_refined = breakout_integrated * fractal_microstructure_score * fractal_eff_momentum
    
    # 10. Fractal Regime Alignment
    fractal_vol_regime = np.where(vol_concentration_fractal > 1.5, 0.8 * fd_3d, 1.0)
    fractal_volatility_regime = np.where(fractal_vol_ratio > 1.2, 1.3 * er_5d, 1.0)
    fractal_regime_multiplier = fractal_vol_regime * fractal_volatility_regime
    
    # Final alpha factor
    final_alpha = microstructure_refined * fractal_regime_multiplier
    
    return final_alpha
