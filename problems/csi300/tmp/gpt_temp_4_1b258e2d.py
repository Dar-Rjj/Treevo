import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Asymmetry & Microstructure Momentum Factor
    Combines intraday price efficiency, volume microstructure patterns, 
    market depth proxies, and multi-timeframe momentum integration
    """
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    # Rolling window for calculations (avoid future data)
    window = 20
    
    for i in range(window, len(df)):
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Intraday price efficiency analysis
        # Opening gap persistence
        gap = (current_data['open'] - current_data['close'].shift(1)).fillna(0)
        gap_fill_rate = (current_data['close'] - current_data['open']) / (gap + 1e-8)
        gap_fill_rate = gap_fill_rate.replace([np.inf, -np.inf], 0)
        
        # Delayed gap reaction momentum (3-day persistence)
        gap_momentum = gap_fill_rate.rolling(window=3, min_periods=1).mean()
        
        # High-Low range utilization
        daily_range = current_data['high'] - current_data['low']
        close_position = (current_data['close'] - current_data['low']) / (daily_range + 1e-8)
        range_efficiency = close_position.rolling(window=5, min_periods=1).std()
        
        # Range expansion momentum
        range_expansion = (daily_range / daily_range.rolling(window=5, min_periods=1).mean() - 1)
        
        # 2. Volume microstructure patterns
        # Early vs late session volume concentration (proxy using rolling periods)
        volume_rolling = current_data['volume'].rolling(window=10, min_periods=1)
        early_volume_ratio = volume_rolling.apply(lambda x: x[:5].sum() / (x.sum() + 1e-8))
        
        # Volume spike persistence
        volume_ma = current_data['volume'].rolling(window=10, min_periods=1).mean()
        volume_spike = (current_data['volume'] / (volume_ma + 1e-8) - 1)
        spike_persistence = volume_spike.rolling(window=3, min_periods=1).mean()
        
        # Tick-level momentum accumulation (proxy using consecutive moves)
        price_changes = current_data['close'].diff()
        consecutive_ups = (price_changes > 0).rolling(window=3).sum()
        consecutive_downs = (price_changes < 0).rolling(window=3).sum()
        directional_consistency = (consecutive_ups - consecutive_downs) / 3.0
        
        # Volume-weighted directional consistency
        vwap = current_data['amount'] / (current_data['volume'] + 1e-8)
        vwap_momentum = (vwap - vwap.rolling(window=5, min_periods=1).mean()) / vwap.rolling(window=5, min_periods=1).std()
        
        # 3. Market depth proxy construction
        # Price impact estimation
        range_volume_ratio = daily_range / (current_data['volume'] + 1e-8)
        latent_pressure = (range_volume_ratio < range_volume_ratio.rolling(window=10).quantile(0.3)).astype(float)
        fragile_levels = (range_volume_ratio > range_volume_ratio.rolling(window=10).quantile(0.7)).astype(float)
        
        # Support/resistance strength (multiple tests of levels)
        recent_highs = current_data['high'].rolling(window=10, min_periods=1).max()
        recent_lows = current_data['low'].rolling(window=10, min_periods=1).min()
        high_tests = (abs(current_data['high'] - recent_highs) / recent_highs < 0.002).rolling(window=5).sum()
        low_tests = (abs(current_data['low'] - recent_lows) / recent_lows < 0.002).rolling(window=5).sum()
        level_strength = (high_tests + low_tests) / 10.0
        
        # 4. Multi-timeframe momentum integration
        # Micro-momentum (intraday)
        intraday_return = (current_data['close'] - current_data['open']) / current_data['open']
        micro_momentum = intraday_return.rolling(window=5, min_periods=1).mean()
        
        # Macro-momentum (multi-day)
        macro_momentum = current_data['close'].pct_change(periods=5).rolling(window=5, min_periods=1).mean()
        
        # Alignment scoring
        momentum_alignment = np.sign(micro_momentum) * np.sign(macro_momentum)
        
        # Divergence detection
        momentum_divergence = abs(micro_momentum - macro_momentum)
        
        # Adaptive holding period (momentum quality)
        momentum_quality = micro_momentum.rolling(window=5, min_periods=1).std()
        signal_decay = 1.0 / (1.0 + abs(micro_momentum))
        
        # Combine all components (current day values only)
        current_idx = current_data.index[-1]
        
        # Weighted combination of factors
        factor = (
            0.15 * gap_momentum.iloc[-1] +
            0.12 * range_efficiency.iloc[-1] +
            0.10 * range_expansion.iloc[-1] +
            0.08 * early_volume_ratio.iloc[-1] +
            0.09 * spike_persistence.iloc[-1] +
            0.11 * directional_consistency.iloc[-1] +
            0.07 * vwap_momentum.iloc[-1] +
            0.06 * latent_pressure.iloc[-1] -
            0.05 * fragile_levels.iloc[-1] +
            0.08 * level_strength.iloc[-1] +
            0.09 * momentum_alignment.iloc[-1] -
            0.06 * momentum_divergence.iloc[-1] +
            0.04 * momentum_quality.iloc[-1] * signal_decay.iloc[-1]
        )
        
        factor_values[current_idx] = factor
    
    # Fill initial NaN values with 0
    factor_values = factor_values.fillna(0)
    
    # Normalize the factor
    if factor_values.std() > 0:
        factor_values = (factor_values - factor_values.mean()) / factor_values.std()
    
    return factor_values
