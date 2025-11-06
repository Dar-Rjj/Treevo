import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price changes and ratios
    data['close_ret_3'] = data['close'] / data['close'].shift(3) - 1
    data['high_low_range'] = data['high'] - data['low']
    data['close_open_diff'] = abs(data['close'] - data['open'])
    data['close_prev_diff'] = abs(data['close'] - data['close'].shift(1))
    data['open_prev_close_diff'] = abs(data['open'] - data['close'].shift(1))
    
    # Hierarchical Gap Volatility Structure
    # Using market average as proxy for index/industry (simplified approach)
    market_avg_close = data['close'].rolling(window=20, min_periods=1).mean()
    market_avg_ret_3 = market_avg_close / market_avg_close.shift(3) - 1
    
    # Stock vs. Index Gap Volatility
    stock_vs_index_gap_vol = (data['close_ret_3'] / (market_avg_ret_3 + 1e-6)) * (data['high_low_range'] / (data['close_prev_diff'] + 1e-6))
    
    # Stock vs. Industry Gap Volatility (using same proxy for simplicity)
    stock_vs_industry_gap_vol = (data['close_ret_3'] / (market_avg_ret_3 + 1e-6)) * (data['high_low_range'] / (data['open_prev_close_diff'] + 1e-6))
    
    # Hierarchical Gap Volatility Divergence
    hierarchical_gap_vol_divergence = stock_vs_index_gap_vol - stock_vs_industry_gap_vol
    
    # Volatility-Scaled Gap Momentum
    gap_momentum_vol_scaling = hierarchical_gap_vol_divergence * (data['high_low_range'] / (abs(data['close'] - data['open']) + 1e-6))
    volume_weighted_gap_momentum = gap_momentum_vol_scaling * (data['volume'] / (data['volume'].shift(3) + 1e-6))
    
    # Hierarchical Gap Volatility Persistence
    gap_vol_sign = np.sign(hierarchical_gap_vol_divergence)
    gap_vol_directional_persistence = gap_vol_sign.rolling(window=5, min_periods=1).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    gap_vol_strength_persistence = gap_vol_directional_persistence * abs(hierarchical_gap_vol_divergence) / (data['high_low_range'] + 1e-6)
    
    # Multi-Timeframe Volatility Regime Analysis
    # Volatility Asymmetry Regime Dynamics
    upside_vol_regime = ((data['high'] - data['close']) / (data['high_low_range'] + 1e-6)) * (abs(data['close'] - data['open']) / (data['high_low_range'] + 1e-6))
    downside_vol_regime = ((data['close'] - data['low']) / (data['high_low_range'] + 1e-6)) * (abs(data['close'] - data['open']) / (data['high_low_range'] + 1e-6))
    volatility_asymmetry_regime_shift = upside_vol_regime - downside_vol_regime * np.sign(hierarchical_gap_vol_divergence)
    
    # Volume Volatility Regime Dynamics
    volume_vol_acceleration = (data['volume'] / (data['volume'].shift(2) + 1e-6) - 1) * np.sign(data['close'] - data['close'].shift(1))
    volume_rolling_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_vol_regime_strength = (np.log(data['volume'] + 1e-6) / np.log(volume_rolling_avg + 1e-6)) * abs((data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-6))
    
    # Position Efficiency Regime Dynamics
    opening_position_efficiency = ((data['close'] - data['open']) / (data['high_low_range'] + 1e-6)) * ((data['open'] - data['close'].shift(1)) / (abs(data['open'] - data['close'].shift(2)) + 1e-6))
    closing_position_efficiency = ((data['high'] - data['close']) / (data['close'] - data['low'] + 1e-6)) * ((data['close'] - data['open']) / (data['high_low_range'] + 1e-6))
    
    # Volatility Momentum Structure with Gap Integration
    # Opening Volatility Momentum
    hierarchical_opening_vol_pressure = ((data['open'] - data['low']) / (data['high'] - data['open'] + 1e-6)) - 1 * np.sign(hierarchical_gap_vol_divergence)
    opening_vol_momentum = (data['open'] - data['close'].shift(1)) * (data['close'] - data['open']) * (abs(data['close'].shift(1) - data['open']) / (data['high_low_range'] + 1e-6))
    
    # Intraday Volatility Momentum
    hierarchical_intraday_vol_flow = hierarchical_gap_vol_divergence * ((data['close'] - data['close'].shift(1)) / (data['high_low_range'] + 1e-6))
    hl2 = (data['high'] + data['low']) / 2
    intraday_vol_momentum = (np.log(data['high_low_range'] + 1e-6) / np.log(abs(data['close'] - hl2) + 1e-6)) * (abs(data['close'] - hl2) / (data['high_low_range'] + 1e-6))
    
    # Closing Volatility Momentum
    hierarchical_closing_vol_efficiency = hierarchical_gap_vol_divergence * (data['volume'] / (data['volume'].shift(1) + 1e-6))
    closing_vol_momentum = (abs(data['close'] - data['open']) / (data['high_low_range'] + 1e-6)) * ((data['close'] - data['close'].shift(1)) - (data['close'].shift(2) - data['close'].shift(3)))
    
    # Dynamic Volatility Regime-Shift Detection
    # Volatility Asymmetry Regime Classification
    vol_asymmetry_regime = pd.Series(index=data.index, dtype=object)
    vol_asymmetry_regime[(upside_vol_regime > 0.6) & (downside_vol_regime < 0.4)] = 'high'
    vol_asymmetry_regime[(upside_vol_regime < 0.3) & (downside_vol_regime > 0.7)] = 'low'
    vol_asymmetry_regime[vol_asymmetry_regime.isna()] = 'normal'
    
    # Volume Volatility Regime Classification
    volume_vol_regime = pd.Series(index=data.index, dtype=object)
    volume_vol_regime[(volume_vol_acceleration > 1.6) & (volume_vol_regime_strength > 1.3)] = 'high'
    volume_vol_regime[(volume_vol_acceleration < 0.6) & (volume_vol_regime_strength < 0.7)] = 'low'
    volume_vol_regime[volume_vol_regime.isna()] = 'normal'
    
    # Position Efficiency Regime Classification
    position_efficiency_regime = pd.Series(index=data.index, dtype=object)
    position_efficiency_regime[(opening_position_efficiency > 0.8) & (closing_position_efficiency > 1.2)] = 'high'
    position_efficiency_regime[(opening_position_efficiency < 0.2) & (closing_position_efficiency < 0.8)] = 'low'
    position_efficiency_regime[position_efficiency_regime.isna()] = 'normal'
    
    # Regime Multipliers
    vol_asymmetry_multiplier = vol_asymmetry_regime.map({'high': 0.5, 'low': 0.15, 'normal': 1.0})
    volume_vol_multiplier = volume_vol_regime.map({'high': 0.35, 'low': 0.08, 'normal': 1.0})
    position_efficiency_multiplier = position_efficiency_regime.map({'high': 0.4, 'low': 0.12, 'normal': 1.0})
    
    combined_vol_regime_multiplier = vol_asymmetry_multiplier * volume_vol_multiplier * position_efficiency_multiplier
    
    # Final Hierarchical Volatility Gap Momentum Alpha
    core_vol_momentum = opening_vol_momentum * intraday_vol_momentum * closing_vol_momentum * hierarchical_gap_vol_divergence
    volume_enhancement = core_vol_momentum * (data['volume'] / (data['volume'].shift(3) + 1e-6)) * gap_vol_strength_persistence
    final_alpha = volume_enhancement * combined_vol_regime_multiplier
    
    return final_alpha
