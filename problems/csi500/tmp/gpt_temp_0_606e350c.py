import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining volatility-scaled momentum acceleration,
    volume-price regime divergence, intraday pattern convergence, and multi-timeframe integration.
    """
    df = data.copy()
    
    # Volatility-Scaled Momentum Acceleration
    # Multi-Timeframe Momentum
    df['mom_3d'] = df['close'] / df['close'].shift(3) - 1
    df['mom_5d'] = df['close'] / df['close'].shift(5) - 1
    df['mom_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Momentum Acceleration
    df['accel_3_5'] = df['mom_5d'] - df['mom_3d']
    df['accel_5_10'] = df['mom_10d'] - df['mom_5d']
    df['accel_persistence'] = np.sign(df['accel_3_5']) * np.sign(df['accel_5_10'])
    
    # Volatility Scaling
    returns = df['close'].pct_change()
    df['vol_3d'] = returns.rolling(window=3).std()
    df['vol_5d'] = returns.rolling(window=5).std()
    df['vol_10d'] = returns.rolling(window=10).std()
    
    # Combined Momentum Signal
    df['geom_mom'] = (df['mom_3d'] * df['mom_5d'] * df['mom_10d']).apply(lambda x: np.sign(x) * (abs(x) ** (1/3)) if x != 0 else 0)
    df['geom_vol'] = (df['vol_3d'] * df['vol_5d'] * df['vol_10d']).apply(lambda x: np.sign(x) * (abs(x) ** (1/3)) if x != 0 else 0)
    momentum_signal = df['geom_mom'] / (df['geom_vol'] + 1e-8) * df['accel_persistence']
    
    # Volume-Price Regime Divergence
    # Price Regime Components
    df['intraday_eff'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['price_persistence'] = np.sign(df['close'] - df['close'].shift(1)) * (df['high'] - df['low']) / (df['close'] + 1e-8)
    df['range_util'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Volume Divergence Signals
    df['vol_accel'] = (df['volume'] / df['volume'].shift(1)) - (df['volume'].shift(1) / df['volume'].shift(2))
    vol_geom_mean = df['volume'].rolling(window=5).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    df['vol_regime'] = df['volume'] / vol_geom_mean
    df['vol_price_corr'] = np.sign(df['close'] - df['close'].shift(1)) * df['volume'] / df['volume'].rolling(window=5).mean()
    
    # Regime Adaptation
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    avg_range = df['daily_range'].rolling(window=5).mean()
    df['high_vol_regime'] = (df['daily_range'] > avg_range).astype(float)
    df['low_vol_regime'] = (df['daily_range'] < avg_range).astype(float)
    df['vol_spike_regime'] = (df['volume'] > 2 * df['volume'].rolling(window=5).mean()).astype(float)
    
    # Combined Volume-Price Signal
    price_components = df[['intraday_eff', 'price_persistence', 'range_util']].replace([np.inf, -np.inf], 0)
    df['core_divergence'] = price_components.apply(lambda x: np.sign(x.prod()) * (abs(x.prod()) ** (1/3)) if x.prod() != 0 else 0, axis=1)
    volume_signal = df['core_divergence'] * df['vol_regime']
    regime_weight = 1 + 0.5 * df['high_vol_regime'] - 0.3 * df['low_vol_regime'] + 0.8 * df['vol_spike_regime']
    volume_signal = volume_signal * regime_weight
    
    # Intraday Pattern Convergence
    # Morning Session Dynamics
    df['open_mom'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    df['morning_strength'] = (df['high'] - df['open']) / (df['open'] + 1e-8)
    df['morning_support'] = (df['open'] - df['low']) / (df['open'] + 1e-8)
    
    # Afternoon Session Dynamics
    df['afternoon_mom'] = (df['close'] - df['high']) / (df['high'] + 1e-8)
    df['afternoon_support'] = (df['close'] - df['low']) / (df['low'] + 1e-8)
    df['closing_eff'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Session Alignment
    df['session_consistency'] = np.sign(df['morning_strength']) * np.sign(df['afternoon_mom'])
    df['intraday_range_eff'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['pressure_balance'] = (df['morning_strength'] - df['morning_support']) * (df['afternoon_mom'] - df['afternoon_support'])
    
    # Combined Intraday Signal
    session_dynamics = df[['open_mom', 'morning_strength', 'morning_support', 'afternoon_mom', 'afternoon_support']].replace([np.inf, -np.inf], 0)
    df['pattern_strength'] = session_dynamics.apply(lambda x: np.sign(x.prod()) * (abs(x.prod()) ** (1/5)) if x.prod() != 0 else 0, axis=1)
    intraday_signal = df['pattern_strength'] * df['session_consistency'] * df['pressure_balance'] * df['closing_eff']
    
    # Multi-Timeframe Geometric Integration
    # Short-term Framework (1-3 days)
    df['st_price_mom'] = df['close'] / df['close'].shift(2) - 1
    df['st_vol_mom'] = df['volume'] / df['volume'].shift(2) - 1
    df['st_range_eff'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low'] + 1e-8)
    
    # Medium-term Framework (5-10 days)
    df['mt_price_trend'] = df['close'] / df['close'].shift(7) - 1
    mt_vol_geom = df['volume'].rolling(window=7).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    df['mt_vol_trend'] = df['volume'] / mt_vol_geom
    mt_range_geom = (df['high'] - df['low']).rolling(window=7).apply(lambda x: np.exp(np.mean(np.log(x + 1e-8))))
    df['mt_vol_regime'] = (df['high'] - df['low']) / mt_range_geom
    
    # Signal Convergence
    df['price_align'] = np.sign(df['st_price_mom']) * np.sign(df['mt_price_trend'])
    df['vol_align'] = np.sign(df['st_vol_mom']) * np.sign(df['mt_vol_trend'])
    df['multi_eff'] = (df['st_range_eff'] * df['mt_vol_regime']).apply(lambda x: np.sign(x) * (abs(x) ** 0.5) if x != 0 else 0)
    
    # Final Integration
    price_factors = df[['st_price_mom', 'mt_price_trend']].replace([np.inf, -np.inf], 0)
    df['core_signal'] = price_factors.apply(lambda x: np.sign(x.prod()) * (abs(x.prod()) ** 0.5) if x.prod() != 0 else 0, axis=1)
    
    vol_factors = df[['st_vol_mom', 'mt_vol_trend']].replace([np.inf, -np.inf], 0)
    df['vol_confirmation'] = vol_factors.apply(lambda x: np.sign(x.prod()) * (abs(x.prod()) ** 0.5) if x.prod() != 0 else 0, axis=1)
    
    multi_signal = df['core_signal'] * df['vol_confirmation'] / (df['mt_vol_regime'] + 1e-8) * df['price_align'] * df['vol_align'] * df['multi_eff']
    
    # Combine all signals with equal weighting
    final_signal = (momentum_signal + volume_signal + intraday_signal + multi_signal) / 4
    
    return final_signal
