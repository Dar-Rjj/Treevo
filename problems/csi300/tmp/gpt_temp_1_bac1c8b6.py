import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Intraday Pressure Imbalance Factor
    high_low_range = df['high'] - df['low']
    high_low_range = high_low_range.replace(0, np.nan)
    
    buying_pressure = ((df['close'] - df['low']) / high_low_range) * df['volume']
    selling_pressure = ((df['high'] - df['close']) / high_low_range) * df['volume']
    
    pressure_ratio = buying_pressure / selling_pressure
    pressure_ratio = pressure_ratio.replace([np.inf, -np.inf], np.nan)
    log_pressure_ratio = np.log(pressure_ratio)
    
    pressure_momentum_3d = log_pressure_ratio.diff(3)
    pressure_momentum_5d = log_pressure_ratio.diff(5)
    
    # Volatility Regime Transition Detector
    short_term_vol = ((df['high'] - df['low']) / df['close']).rolling(3).std()
    medium_term_vol = ((df['high'] - df['low']) / df['close']).rolling(10).std()
    
    vol_ratio = short_term_vol / medium_term_vol
    vol_ratio_momentum = vol_ratio.diff(3)
    
    volume_ratio = df['volume'] / df['volume'].rolling(5).mean()
    vol_transition_signal = vol_ratio_momentum * volume_ratio
    
    # Opening Gap Momentum Persistence
    prev_close = df['close'].shift(1)
    gap_strength = (df['open'] - prev_close) / prev_close
    abs_gap = gap_strength.abs()
    
    gap_persistence = (df['close'] - df['open']) / high_low_range
    gap_direction_aligned = gap_persistence * np.sign(gap_strength)
    
    gap_momentum_2d = gap_direction_aligned.rolling(2).mean()
    gap_momentum_5d = gap_direction_aligned.rolling(5).mean()
    
    volume_validation = df['volume'] / df['volume'].rolling(10).mean()
    gap_signal = (gap_momentum_2d + gap_momentum_5d) * volume_validation
    
    # Microstructure Efficiency Ratio
    true_range = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    up_movement = df['high'] - df['high'].shift(1)
    down_movement = df['low'].shift(1) - df['low']
    directional_movement = up_movement.clip(lower=0) + down_movement.clip(lower=0)
    
    efficiency_ratio = directional_movement / true_range.replace(0, np.nan)
    rolling_efficiency = efficiency_ratio.rolling(5).mean()
    
    volume_efficiency = df['volume'] / df['amount'].replace(0, np.nan)
    combined_efficiency = rolling_efficiency * volume_efficiency
    
    # Cumulative Order Flow Imbalance
    daily_order_flow = ((2 * df['close'] - df['high'] - df['low']) / high_low_range) * df['volume']
    cum_flow_5d = daily_order_flow.rolling(5).sum()
    cum_flow_10d = daily_order_flow.rolling(10).sum()
    
    flow_ratio = cum_flow_5d / cum_flow_10d.replace(0, np.nan)
    flow_acceleration = flow_ratio.diff(3)
    
    price_momentum = df['close'].pct_change(3)
    flow_divergence = flow_acceleration - price_momentum
    
    # Intraday Reversal Strength
    morning_strength = ((df['high'] - df['open']) / high_low_range) * (df['volume'] * 0.6)
    afternoon_strength = ((df['close'] - df['low']) / high_low_range) * (df['volume'] * 0.4)
    
    session_imbalance = morning_strength - afternoon_strength
    session_momentum = session_imbalance.diff(1)
    
    # Pressure Decay Momentum
    daily_pressure = ((df['close'] - df['open']) / high_low_range) * df['volume']
    pressure_momentum_3d = daily_pressure.diff(3)
    pressure_decay_5d = daily_pressure.rolling(5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x))
    
    # Regime-Sensitive Momentum
    vol_5d = df['close'].pct_change().rolling(5).std()
    vol_10d = df['close'].pct_change().rolling(10).std()
    
    high_vol_regime = (vol_5d > vol_10d * 1.5) & (df['volume'] > df['volume'].rolling(10).mean())
    low_vol_regime = (vol_5d < vol_10d * 0.7) & (df['volume'] <= df['volume'].rolling(10).mean() * 1.2)
    
    short_momentum = df['close'].pct_change(3)
    medium_momentum = df['close'].pct_change(8)
    
    regime_momentum = pd.Series(np.nan, index=df.index)
    regime_momentum[high_vol_regime] = short_momentum[high_vol_regime]
    regime_momentum[low_vol_regime] = medium_momentum[low_vol_regime]
    
    vol_adjusted_momentum = regime_momentum / (vol_5d.replace(0, np.nan) + 0.001)
    volume_weighted_signal = vol_adjusted_momentum * (df['volume'] / df['volume'].rolling(10).mean())
    
    # Microstructure Noise Persistence
    noise_component = high_low_range / (df['close'] - df['open']).abs().replace(0, np.nan)
    volume_adjusted_noise = noise_component * df['volume']
    
    noise_autocorr = volume_adjusted_noise.rolling(3).apply(lambda x: x.autocorr(lag=1))
    noise_trend = volume_adjusted_noise.rolling(5).apply(lambda x: (x.iloc[-1] - x.iloc[0]) / len(x))
    
    # Multi-Timeframe Pressure Convergence
    intraday_pressure = ((df['close'] - df['open']) / high_low_range) * df['volume']
    short_term_pressure = intraday_pressure.rolling(3).sum()
    short_term_momentum = short_term_pressure.diff(3)
    
    medium_term_pressure = intraday_pressure.rolling(8).sum()
    medium_term_acceleration = medium_term_pressure.diff(5)
    
    # Combine factors with weights
    factor = (
        0.15 * pressure_momentum_3d +
        0.12 * pressure_momentum_5d +
        0.10 * vol_transition_signal +
        0.08 * gap_signal +
        0.09 * combined_efficiency +
        0.11 * flow_divergence +
        0.07 * session_momentum +
        0.08 * pressure_decay_5d +
        0.12 * volume_weighted_signal +
        0.08 * noise_trend
    )
    
    return factor
