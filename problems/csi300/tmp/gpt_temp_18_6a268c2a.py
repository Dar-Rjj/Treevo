import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    intraday_pressure = (df['high'] - df['open']) - (df['open'] - df['low'])
    
    # Movement Efficiency calculation
    prev_close = df['close'].shift(1)
    close_move = abs(df['close'] - prev_close)
    high_low_range = df['high'] - df['low']
    high_prev_range = abs(df['high'] - prev_close)
    low_prev_range = abs(df['low'] - prev_close)
    movement_efficiency = close_move / np.maximum(high_low_range, np.maximum(high_prev_range, low_prev_range))
    
    # Efficiency-Weighted Imbalance
    efficiency_weighted_imbalance = intraday_pressure * movement_efficiency
    
    # Multi-Scale Microstructure Momentum
    ewi_5d = efficiency_weighted_imbalance.rolling(window=5, min_periods=3).sum()
    ewi_10d = efficiency_weighted_imbalance.rolling(window=10, min_periods=5).sum()
    microstructure_flow_momentum = ewi_5d - ewi_10d
    
    # Tick-Level Consistency (5-day correlation)
    tick_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window_data = df.iloc[i-4:i+1]
            if len(window_data) >= 3:
                corr = np.corrcoef(window_data['high'] - window_data['open'] - (window_data['open'] - window_data['low']), 
                                 movement_efficiency.iloc[i-4:i+1])[0,1]
                tick_consistency.iloc[i] = corr if not np.isnan(corr) else 0
            else:
                tick_consistency.iloc[i] = 0
        else:
            tick_consistency.iloc[i] = 0
    
    daily_microstructure_alignment = np.sign(microstructure_flow_momentum) * tick_consistency
    microstructure_momentum_signal = microstructure_flow_momentum * daily_microstructure_alignment
    
    # Volatility-Regime Microstructure Quality
    microstructure_vol_5d = abs(efficiency_weighted_imbalance).rolling(window=5, min_periods=3).sum()
    microstructure_vol_10d = abs(efficiency_weighted_imbalance).rolling(window=10, min_periods=5).sum()
    microstructure_compression = microstructure_vol_5d / microstructure_vol_10d - 1
    
    # Realized volatility for regime detection
    returns = df['close'].pct_change()
    realized_vol_5d = returns.rolling(window=5, min_periods=3).std()
    realized_vol_10d = returns.rolling(window=10, min_periods=5).std()
    
    regime_weighted_microstructure = microstructure_momentum_signal * (1 + microstructure_compression)
    
    # Volume breakout confirmation
    volume_10d_avg = df['volume'].rolling(window=10, min_periods=5).mean().shift(1)
    volume_breakout = df['volume'] > (2.5 * volume_10d_avg)
    efficiency_breakout = movement_efficiency > 0.8
    quality_breakout_signal = regime_weighted_microstructure * volume_breakout * efficiency_breakout
    
    # Price Discovery Efficiency Momentum
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    discovery_momentum = (df['close'] - df['close'].shift(5)) * (intraday_efficiency + movement_efficiency)
    information_velocity = discovery_momentum * np.sign(intraday_pressure)
    
    # 5-day Discovery Consistency
    discovery_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window_data = df.iloc[i-4:i+1]
            if len(window_data) >= 3:
                intraday_eff_window = (window_data['close'] - window_data['open']) / (window_data['high'] - window_data['low']).replace(0, np.nan)
                movement_eff_window = movement_efficiency.iloc[i-4:i+1]
                valid_mask = ~(intraday_eff_window.isna() | movement_eff_window.isna())
                if valid_mask.sum() >= 2:
                    corr = np.corrcoef(intraday_eff_window[valid_mask], movement_eff_window[valid_mask])[0,1]
                    discovery_consistency.iloc[i] = corr if not np.isnan(corr) else 0
                else:
                    discovery_consistency.iloc[i] = 0
            else:
                discovery_consistency.iloc[i] = 0
        else:
            discovery_consistency.iloc[i] = 0
    
    discovery_quality = information_velocity * discovery_consistency
    adaptive_discovery_signal = discovery_quality * microstructure_compression
    
    info_velocity_10d_avg = abs(information_velocity).rolling(window=10, min_periods=5).mean().shift(1)
    high_discovery_regime = abs(information_velocity) > info_velocity_10d_avg
    regime_enhanced_discovery = adaptive_discovery_signal * (1 + np.sign(information_velocity))
    
    # Volume-Amount Microstructure Confluence
    volume_price_corr = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:
            window_data = df.iloc[i-4:i+1]
            if len(window_data) >= 3:
                corr = np.corrcoef(window_data['close'], window_data['volume'])[0,1]
                volume_price_corr.iloc[i] = corr if not np.isnan(corr) else 0
            else:
                volume_price_corr.iloc[i] = 0
        else:
            volume_price_corr.iloc[i] = 0
    
    amount_5d_avg = df['amount'].rolling(window=5, min_periods=3).mean()
    amount_weighted_pressure = intraday_pressure * (df['amount'] / amount_5d_avg.replace(0, np.nan))
    efficient_informed_flow = amount_weighted_pressure * volume_price_corr * movement_efficiency
    
    eif_5d = efficient_informed_flow.rolling(window=5, min_periods=3).sum()
    eif_10d = efficient_informed_flow.rolling(window=10, min_periods=5).sum()
    information_resolution_momentum = eif_5d - eif_10d
    
    microstructure_volume_alignment = information_resolution_momentum * volume_price_corr
    efficiency_confirmation = microstructure_volume_alignment * movement_efficiency
    final_confluence_signal = efficiency_confirmation * np.sign(discovery_quality)
    
    # Adaptive Alpha Synthesis
    # Volatility Regime Classification
    high_vol_regime = realized_vol_5d > (1.5 * realized_vol_10d)
    low_vol_regime = realized_vol_5d < (0.7 * realized_vol_10d)
    normal_vol_regime = ~(high_vol_regime | low_vol_regime)
    
    # Efficiency Regime Classification
    strong_efficiency = movement_efficiency > 0.8
    weak_efficiency = movement_efficiency < 0.3
    normal_efficiency = ~(strong_efficiency | weak_efficiency)
    
    # Microstructure Regime Classification
    microstructure_signal_10d_avg = abs(microstructure_momentum_signal).rolling(window=10, min_periods=5).mean().shift(1)
    high_microstructure = abs(microstructure_momentum_signal) > microstructure_signal_10d_avg
    low_microstructure = abs(microstructure_momentum_signal) < microstructure_signal_10d_avg
    normal_microstructure = ~(high_microstructure | low_microstructure)
    
    # Component Factors
    microstructure_factor = regime_weighted_microstructure
    discovery_factor = regime_enhanced_discovery
    confluence_factor = final_confluence_signal
    quality_factor = quality_breakout_signal
    
    # Volatility-Weighted Combination
    vol_weights = pd.Series(1.0, index=df.index)
    vol_weights[high_vol_regime] = 0.7
    vol_weights[low_vol_regime] = 1.3
    vol_weights[normal_vol_regime] = 1.0
    
    volatility_weighted_combination = (
        microstructure_factor * vol_weights +
        discovery_factor * vol_weights +
        confluence_factor * vol_weights +
        quality_factor * vol_weights
    ) / 4
    
    # Efficiency-Weighted Combination
    eff_weights = pd.Series(1.0, index=df.index)
    eff_weights[strong_efficiency] = 1.2
    eff_weights[weak_efficiency] = 0.8
    eff_weights[normal_efficiency] = 1.0
    
    efficiency_weighted_combination = (
        microstructure_factor * eff_weights +
        discovery_factor * eff_weights +
        confluence_factor * eff_weights +
        quality_factor * eff_weights
    ) / 4
    
    # Final Alpha
    final_alpha = (volatility_weighted_combination + efficiency_weighted_combination) / 2
    
    return final_alpha
