import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility Transmission with Microstructure Convergence alpha factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    vol_std_10 = data['close'].rolling(window=10, min_periods=10).std()
    vol_std_20 = data['close'].rolling(window=20, min_periods=20).std()
    volatility_ratio = vol_std_10 / vol_std_20
    
    # Regime States
    high_regime = (volatility_ratio > 1.2).astype(int)
    low_regime = (volatility_ratio < 0.8).astype(int)
    normal_regime = ((volatility_ratio >= 0.8) & (volatility_ratio <= 1.2)).astype(int)
    
    # Multi-Timeframe Volatility Transmission
    
    # Short-Term Transmission (3-day)
    opening_momentum = (data['open'] / data['close'].shift(1) - 1) * volatility_ratio
    intraday_transmission = (data['high'] - data['low']) / data['close'].rolling(window=5, min_periods=5).std()
    
    # Volatility persistence: consecutive same-direction volatility days weighted by volume
    vol_direction = np.sign(data['close'].pct_change())
    vol_persistence = vol_direction.rolling(window=3).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and not pd.isna(x.iloc[i]) and not pd.isna(x.iloc[i-1])]), 
        raw=False
    )
    volatility_persistence = vol_persistence * (data['volume'] / data['volume'].rolling(window=20, min_periods=20).mean())
    
    short_term_transmission = opening_momentum + intraday_transmission + volatility_persistence
    
    # Medium-Term Transmission (8-day)
    gap_dynamics = abs(data['open'] / data['close'].shift(1) - 1) * volatility_ratio
    
    # Transmission persistence: consecutive transmission days
    transmission_direction = np.sign(short_term_transmission)
    transmission_persistence = transmission_direction.rolling(window=8).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and not pd.isna(x.iloc[i]) and not pd.isna(x.iloc[i-1])]), 
        raw=False
    )
    
    # Volatility trend strength
    high_7d = data['high'].rolling(window=7, min_periods=7).max()
    low_7d = data['low'].rolling(window=7, min_periods=7).min()
    volatility_trend_strength = (high_7d / low_7d - 1) * volatility_ratio
    
    medium_term_transmission = gap_dynamics + transmission_persistence + volatility_trend_strength
    
    # Regime-Adaptive Transmission
    transmission_components = short_term_transmission + medium_term_transmission
    regime_adaptive_transmission = (
        high_regime * transmission_components * volatility_ratio +
        low_regime * transmission_components / volatility_ratio +
        normal_regime * transmission_components
    )
    
    # Microstructure Convergence Detection
    
    # Price-Microstructure Convergence
    directional_efficiency = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    gap_volatility = data['open'].pct_change().abs().rolling(window=5, min_periods=5).std()
    gap_filling_efficiency = directional_efficiency * gap_volatility
    
    # Price momentum stability
    returns_5d = data['close'].pct_change(periods=5)
    price_momentum_stability = returns_5d.rolling(window=5, min_periods=5).std() * (data['volume'] / data['volume'].rolling(window=20, min_periods=20).mean())
    
    price_convergence = directional_efficiency + gap_filling_efficiency - price_momentum_stability
    
    # Volume-Microstructure Convergence
    volume_mean_20 = data['volume'].rolling(window=20, min_periods=20).mean()
    large_trades = data['volume'] > volume_mean_20
    large_trade_dominance = (
        data['volume'].rolling(window=5, min_periods=5).apply(
            lambda x: x[large_trades.loc[x.index].values].sum() if large_trades.loc[x.index].any() else 0, 
            raw=False
        ) / data['volume'].rolling(window=5, min_periods=5).sum()
    )
    
    volume_intensity = (data['volume'] / data['volume'].rolling(window=5, min_periods=5).mean()) * large_trade_dominance
    volume_volatility_ratio = data['volume'].rolling(window=10, min_periods=10).std() / data['volume'].rolling(window=10, min_periods=10).mean()
    
    volume_convergence = large_trade_dominance + volume_intensity - volume_volatility_ratio
    
    # Convergence Quality Assessment
    convergence_quality = (price_convergence + volume_convergence) / 2
    
    # Volatility-Microstructure Transmission Engine
    transmission_momentum = regime_adaptive_transmission.rolling(window=5, min_periods=5).mean()
    microstructure_efficiency = convergence_quality.rolling(window=5, min_periods=5).mean()
    
    volatility_microstructure_alignment = transmission_momentum * microstructure_efficiency
    
    # Multi-timeframe alignment
    short_term_alignment = short_term_transmission.rolling(window=3, min_periods=3).mean() * microstructure_efficiency
    medium_term_alignment = medium_term_transmission.rolling(window=8, min_periods=8).mean() * microstructure_efficiency
    multi_timeframe_alignment = (short_term_alignment + medium_term_alignment) / 2
    
    # Regime-adaptive alignment
    regime_adaptive_alignment = (
        high_regime * volatility_microstructure_alignment * volatility_ratio +
        low_regime * volatility_microstructure_alignment / volatility_ratio +
        normal_regime * volatility_microstructure_alignment
    )
    
    # Breakout Confirmation with Microstructure Transmission
    
    # Price Breakout Detection
    high_5d_max = data['high'].rolling(window=5, min_periods=5).max()
    relative_strength = (data['close'] - high_5d_max.shift(1)) / high_5d_max.shift(1)
    
    # Breakout persistence
    breakout_signal = (data['close'] > high_5d_max.shift(1)).astype(int)
    breakout_persistence = breakout_signal.rolling(window=5, min_periods=5).sum()
    
    price_breakout = relative_strength * breakout_persistence * large_trade_dominance
    
    # Volume-Microstructure Breakout
    volume_surge = (data['volume'] / data['volume'].rolling(window=5, min_periods=5).mean()) * large_trade_dominance
    volume_breakout_alignment = volume_surge * price_breakout
    volume_breakout_reliability = volume_breakout_alignment * convergence_quality
    
    # Combined Transmission Breakout
    combined_breakout = (price_breakout + volume_breakout_reliability) * volatility_microstructure_alignment
    
    # Composite Alpha Construction
    
    # Core Transmission Component
    multi_timeframe_transmission = (short_term_transmission + medium_term_transmission) / 2
    transmission_persistence_weight = transmission_persistence / transmission_persistence.rolling(window=20, min_periods=20).max()
    core_transmission = multi_timeframe_transmission * transmission_persistence_weight
    
    # Microstructure Convergence Enhancement
    price_convergence_score = price_convergence.rolling(window=5, min_periods=5).mean()
    volume_convergence_quality = volume_convergence.rolling(window=5, min_periods=5).mean()
    
    # Convergence persistence
    convergence_signal = (convergence_quality > convergence_quality.rolling(window=20, min_periods=20).mean()).astype(int)
    convergence_persistence = convergence_signal.rolling(window=5, min_periods=5).sum()
    convergence_multiplier = 1 + (convergence_persistence / 5)
    
    microstructure_enhancement = (price_convergence_score + volume_convergence_quality) * convergence_multiplier
    
    # Breakout Confirmation Overlay
    breakout_confirmation = combined_breakout * microstructure_enhancement
    
    # Final Alpha Signal
    alpha_signal = (
        core_transmission * 0.4 +
        microstructure_enhancement * 0.3 +
        breakout_confirmation * 0.3
    )
    
    # Regime-adaptive final signal
    final_alpha = (
        high_regime * alpha_signal * volatility_ratio +
        low_regime * alpha_signal / volatility_ratio +
        normal_regime * alpha_signal
    )
    
    return final_alpha
