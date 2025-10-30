import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Horizon Momentum Elasticity Analysis
    df = df.copy()
    
    # Calculate momentum across different timeframes
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_8d'] = df['close'] / df['close'].shift(8) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Quantify momentum elasticity (divergence across timeframes)
    df['momentum_divergence'] = (df['momentum_3d'] - df['momentum_8d']) + (df['momentum_8d'] - df['momentum_20d'])
    df['momentum_acceleration'] = (df['momentum_3d'] - df['momentum_3d'].shift(3)) - (df['momentum_20d'] - df['momentum_20d'].shift(3))
    
    # Volume-Price Asymmetry Detection
    df['volume_20d_avg'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['volume_shock_ratio'] = df['volume'] / df['volume_20d_avg']
    
    # Volume cluster intensity (5-day volume persistence)
    df['volume_persistence'] = df['volume'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and np.std(x) > 0 else 0
    )
    
    # Price-Volume Asymmetry Assessment
    df['daily_return'] = df['close'] / df['close'].shift(1) - 1
    
    # Calculate up-day and down-day volume characteristics
    up_days_mask = df['daily_return'] > 0
    down_days_mask = df['daily_return'] < 0
    
    # Rolling average volume on up days and down days
    df['up_day_volume_10d'] = df['volume'].where(up_days_mask).rolling(window=10, min_periods=5).mean()
    df['down_day_volume_10d'] = df['volume'].where(down_days_mask).rolling(window=10, min_periods=5).mean()
    
    # Volume Asymmetry Ratio
    df['volume_asymmetry'] = df['up_day_volume_10d'] / df['down_day_volume_10d']
    df['volume_asymmetry'] = df['volume_asymmetry'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    # Intraday Efficiency & Reversal Analysis
    df['intraday_strength'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['intraday_strength_persistence'] = df['intraday_strength'].rolling(window=5).std()
    
    # Opening Gap Analysis
    df['opening_gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['gap_reversion'] = (df['close'] - df['open']) / (df['open'] - df['close'].shift(1)).replace(0, np.nan)
    
    # Failed Breakout Detection
    df['failed_breakout'] = ((df['opening_gap'].abs() > df['opening_gap'].rolling(window=20).std()) & 
                            (df['gap_reversion'] < -0.5)).astype(int)
    
    # Liquidity-Weighted Regime Adaptation
    df['liquidity_score'] = df['volume'] * df['amount']
    df['liquidity_persistence'] = df['liquidity_score'].rolling(window=5).std()
    
    # Price Level Context
    df['price_position'] = (df['close'] - df['low'].rolling(window=20).min()) / \
                          (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min())
    df['recent_volatility'] = (df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()) / df['close'].rolling(window=20).mean()
    
    # Adaptive Signal Synthesis
    # Combine momentum elasticity with volume asymmetry
    df['momentum_volume_signal'] = df['momentum_divergence'] * df['volume_asymmetry']
    
    # Apply directional consistency weighting
    momentum_direction = np.sign(df['momentum_3d'])
    df['directional_signal'] = df['momentum_volume_signal'] * momentum_direction
    
    # Enhance with intraday context
    df['intraday_enhanced'] = df['directional_signal'] * (1 - df['failed_breakout']) * (1 - df['intraday_strength_persistence'])
    
    # Regime-adaptive fusion
    liquidity_weight = np.tanh(df['liquidity_score'] / df['liquidity_score'].rolling(window=20).mean())
    volatility_scaling = 1 / (1 + df['recent_volatility'])
    price_level_adjustment = 1 + 0.5 * np.sin(np.pi * df['price_position'])
    
    df['regime_adjusted'] = df['intraday_enhanced'] * liquidity_weight * volatility_scaling * price_level_adjustment
    
    # Persistence validation
    df['signal_persistence'] = df['regime_adjusted'].rolling(window=3).apply(
        lambda x: 1 if len(x) == 3 and np.all(np.diff(np.sign(x)) == 0) else 0.5
    )
    
    # Final alpha factor
    df['alpha_factor'] = df['regime_adjusted'] * df['signal_persistence']
    
    # Clean and return
    alpha_series = df['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha_series
