import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Quality
    # Direction alignment across 3/5/10-day returns
    ret_3d = data['close'].pct_change(3)
    ret_5d = data['close'].pct_change(5)
    ret_10d = data['close'].pct_change(10)
    
    # Momentum acceleration
    mom_acceleration = (ret_5d - ret_10d) * np.sign(ret_5d)
    
    # Volume-weighted return persistence
    vol_weighted_ret_3d = (data['close'].pct_change(3) * data['volume'].rolling(3).mean()).fillna(0)
    vol_weighted_ret_5d = (data['close'].pct_change(5) * data['volume'].rolling(5).mean()).fillna(0)
    vol_weighted_ret_10d = (data['close'].pct_change(10) * data['volume'].rolling(10).mean()).fillna(0)
    
    momentum_quality = (
        np.sign(ret_3d) * np.sign(ret_5d) * np.sign(ret_10d) +  # Direction alignment
        mom_acceleration.rank(pct=True) +  # Momentum acceleration
        (vol_weighted_ret_3d.rank(pct=True) + vol_weighted_ret_5d.rank(pct=True) + vol_weighted_ret_10d.rank(pct=True)) / 3
    )
    
    # Microstructure Efficiency Assessment
    # Intraday range efficiency consistency
    intraday_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    range_efficiency_consistency = intraday_efficiency.rolling(5).std().fillna(0)
    
    # Gap utilization
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    same_day_return = (data['close'] - data['open']) / data['open']
    gap_utilization = overnight_gap * np.sign(same_day_return)
    
    # Session pressure asymmetry (using volume patterns)
    morning_volume_ratio = data['volume'].rolling(5).apply(
        lambda x: x.iloc[:len(x)//2].sum() / (x.sum() + 1e-8) if len(x) > 0 else 0.5, raw=False
    )
    session_pressure = (morning_volume_ratio - 0.5).abs().rolling(3).mean()
    
    microstructure_efficiency = (
        (1 - range_efficiency_consistency.rank(pct=True)) +  # Higher consistency is better
        gap_utilization.rank(pct=True) +
        (1 - session_pressure.rank(pct=True))  # Lower pressure asymmetry is better
    )
    
    # Liquidity-Momentum Convergence
    # Volume concentration momentum
    volume_rolling = data['volume'].rolling(10)
    top_30_volume_ratio = volume_rolling.apply(
        lambda x: np.mean(np.sort(x)[-int(len(x)*0.3):]) / (np.mean(x) + 1e-8) if len(x) > 0 else 1, raw=False
    )
    volume_concentration_trend = top_30_volume_ratio.pct_change(3).fillna(0)
    
    # Order flow imbalance momentum (using amount/volume as proxy)
    ofi_proxy = (data['amount'] / (data['volume'] + 1e-8)).pct_change(3)
    ofi_momentum = ofi_proxy.rolling(5).mean().fillna(0)
    
    # Market depth momentum (using high-low range as depth proxy)
    depth_proxy = (data['high'] - data['low']).rolling(5).mean()
    depth_momentum = depth_proxy.pct_change(3).fillna(0)
    
    liquidity_momentum = (
        volume_concentration_trend.rank(pct=True) +
        ofi_momentum.rank(pct=True) +
        depth_momentum.rank(pct=True)
    )
    
    # Cross-Timeframe Signal Integration
    # Ultra-short vs short-term momentum divergence
    ultra_short_mom = data['close'].pct_change(1)
    short_term_mom = data['close'].pct_change(5)
    momentum_divergence = (ultra_short_mom - short_term_mom).abs().rolling(3).mean()
    
    # Multi-timeframe weighted momentum alignment
    weights = np.array([0.4, 0.35, 0.25])  # Higher weight to shorter timeframes
    multi_timeframe_mom = (
        weights[0] * ultra_short_mom.rank(pct=True) +
        weights[1] * ret_5d.rank(pct=True) +
        weights[2] * ret_10d.rank(pct=True)
    )
    
    # Momentum-pressure regime adaptation
    volatility_regime = data['close'].pct_change().rolling(10).std().fillna(0)
    regime_adaptive_momentum = multi_timeframe_mom / (volatility_regime + 1e-8)
    
    cross_timeframe_signal = (
        (1 - momentum_divergence.rank(pct=True)) +  # Lower divergence is better
        regime_adaptive_momentum.rank(pct=True)
    )
    
    # Risk & Divergence Monitoring
    # Momentum-liquidity decoupling detection
    momentum_liquidity_corr = data['close'].pct_change(5).rolling(10).corr(data['volume'].pct_change(5)).fillna(0)
    decoupling_risk = (1 - momentum_liquidity_corr.abs()).rolling(3).mean()
    
    # Gap reversal risk assessment
    gap_reversal_risk = (overnight_gap * data['close'].pct_change(1)).rolling(5).std().fillna(0)
    
    # Cross-timeframe momentum divergence alerts
    timeframe_divergence = (
        (ret_3d.rank(pct=True) - ret_5d.rank(pct=True)).abs() +
        (ret_5d.rank(pct=True) - ret_10d.rank(pct=True)).abs()
    ).rolling(3).mean()
    
    risk_monitoring = (
        (1 - decoupling_risk.rank(pct=True)) +
        (1 - gap_reversal_risk.rank(pct=True)) +
        (1 - timeframe_divergence.rank(pct=True))
    )
    
    # Final factor integration with equal weighting
    final_factor = (
        momentum_quality.fillna(0) +
        microstructure_efficiency.fillna(0) +
        liquidity_momentum.fillna(0) +
        cross_timeframe_signal.fillna(0) +
        risk_monitoring.fillna(0)
    )
    
    return final_factor
