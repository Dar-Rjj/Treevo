import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Weighted Momentum Convergence
    # Multi-Timeframe Momentum
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_8d'] = df['close'] / df['close'].shift(8) - 1
    df['momentum_13d'] = df['close'] / df['close'].shift(13) - 1
    
    # Momentum Convergence
    df['momentum_convergence'] = df['momentum_3d'] - df['momentum_8d']
    
    # Volatility Adjustment
    df['high_low_range'] = df['high'] - df['low']
    df['volatility_10d'] = df['high_low_range'].rolling(window=10).mean()
    
    # Signal Generation
    volatility_weighted_momentum = df['momentum_convergence'] / (df['volatility_10d'] + 1e-8)
    
    # Volume-Confirmed Price Acceleration
    # Price Acceleration
    df['return_2d'] = df['close'] / df['close'].shift(2) - 1
    df['return_5d'] = df['close'] / df['close'].shift(5) - 1
    df['price_acceleration'] = df['return_5d'] - df['return_2d']
    
    # Volume Confirmation
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_5d_avg'] + 1e-8)
    df['volume_deviation'] = (df['volume'] - df['volume_5d_avg']) / (df['volume_5d_avg'] + 1e-8)
    
    # Signal Combination
    volume_confirmed_acceleration = df['price_acceleration'] * df['volume_ratio']
    
    # Relative Strength Breakout with Volatility Filter
    # Relative Strength
    df['high_20d'] = df['high'].rolling(window=20).max()
    df['relative_strength'] = (df['close'] - df['close'].shift(20)) / (df['high_20d'] - df['close'].shift(20) + 1e-8)
    
    # Volume Spike
    df['volume_80th'] = df['volume'].rolling(window=20).quantile(0.8)
    df['volume_spike'] = (df['volume'] > df['volume_80th']).astype(float)
    
    # Volatility Filter
    df['volatility_filter'] = df['high_low_range'].rolling(window=10).mean()
    
    # Signal Generation
    strength_breakout = df['relative_strength'] * df['volume_spike'] / (df['volatility_filter'] + 1e-8)
    
    # Intraday Pressure with Volume Persistence
    # Intraday Pressure
    df['intraday_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Volume Persistence
    df['volume_above_avg'] = (df['volume'] > df['volume_5d_avg']).astype(int)
    df['volume_persistence'] = df['volume_above_avg'].rolling(window=5, min_periods=1).sum()
    
    # Signal Combination
    intraday_pressure_signal = df['intraday_pressure'] * df['volume_persistence']
    
    # Efficiency-Weighted Turnover
    # Price Efficiency
    df['price_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high_low_range'] + 1e-8)
    
    # Turnover Rate
    df['turnover_rate'] = df['volume'] / (df['amount'] + 1e-8)
    
    # Efficiency Adjustment
    efficiency_turnover = df['turnover_rate'] * df['price_efficiency']
    
    # Gap Filling with Momentum Confirmation
    # Price Gap
    df['price_gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
    
    # Momentum Confirmation
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    
    # Volume Trend
    df['volume_trend'] = df['volume_ratio']
    
    # Signal Generation
    gap_filling_signal = df['price_gap'] * df['momentum_5d'] * df['volume_trend']
    
    # Volatility-Regime Momentum Divergence
    # Momentum Divergence
    df['return_5d_alt'] = df['close'] / df['close'].shift(5) - 1
    df['return_10d'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_divergence'] = df['return_5d_alt'] - df['return_10d']
    
    # Volatility Regime
    df['volatility_regime'] = df['volatility_10d'] / df['volatility_10d'].rolling(window=20).mean()
    
    # Signal Generation
    volatility_regime_divergence = df['momentum_divergence'] * df['volatility_regime']
    
    # Convergence-Divergence with Volume Validation
    # Multi-Timeframe Convergence
    df['momentum_3d_alt'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_8d_alt'] = df['close'] / df['close'].shift(8) - 1
    df['convergence_pattern'] = (df['momentum_3d_alt'] > df['momentum_8d_alt']).astype(float)
    
    # Volume Validation
    df['volume_trend_5d'] = df['volume'].rolling(window=5).apply(lambda x: 1 if (x.diff().dropna() > 0).all() else 0, raw=False)
    
    # Signal Generation
    convergence_volume_signal = df['convergence_pattern'] * df['volume_trend_5d']
    
    # Combine all signals with equal weights
    final_signal = (
        volatility_weighted_momentum +
        volume_confirmed_acceleration +
        strength_breakout +
        intraday_pressure_signal +
        efficiency_turnover +
        gap_filling_signal +
        volatility_regime_divergence +
        convergence_volume_signal
    ) / 8
    
    return final_signal
