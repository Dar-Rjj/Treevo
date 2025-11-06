import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate True Range
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = abs(df['high'] - df['prev_close'])
    df['tr3'] = abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Range Efficiency components
    df['daily_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['efficiency_5ma'] = df['daily_efficiency'].rolling(window=5).mean()
    df['efficiency_slope'] = df['daily_efficiency'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / 4 if len(x) == 5 else np.nan)
    
    # Liquidity Dynamics
    df['volume_momentum'] = df['volume'] / df['volume'].shift(1) - 1
    df['large_trade_ratio'] = df['amount'] / df['volume']  # Proxy for large trade impact
    
    # Market State
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1)) - 1
    df['efficiency_regime'] = df['daily_efficiency'] - df['efficiency_5ma']
    
    # Volatility Patterns
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['volatility_ratio'] = df['true_range'] / df['atr_5']
    df['volatility_slope'] = df['true_range'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / 4 if len(x) == 5 else np.nan)
    
    # Momentum Dynamics
    df['short_momentum'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['medium_momentum'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    df['momentum_convergence'] = df['short_momentum'] - df['medium_momentum']
    
    # Intraday Patterns
    df['morning_strength'] = (df['high'] - df['open']) / df['open']
    df['afternoon_support'] = (df['close'] - df['low']) / df['low']
    df['morning_efficiency'] = (df['high'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['afternoon_efficiency'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Alpha Signals from Quantum Range-Liquidity Momentum
    efficiency_momentum = df['daily_efficiency'] * df['volume_momentum']
    liquidity_range_synergy = df['large_trade_ratio'] * df['range_expansion']
    
    # Alpha Signals from Volatility-Momentum Regime Shift
    volatility_weighted_momentum = df['momentum_convergence'] * df['volatility_ratio']
    volatility_breakout = (df['volatility_ratio'] > 1.5).astype(float)
    regime_shift_signal = volatility_breakout * df['momentum_convergence']
    
    # Alpha Signals from Intraday Efficiency-Liquidity
    intraday_momentum = df['morning_strength'] * df['large_trade_ratio']
    efficiency_liquidity_alpha = df['afternoon_efficiency'] * df['large_trade_ratio']
    
    # Combine alpha signals with weights
    alpha = (
        0.3 * efficiency_momentum +
        0.25 * liquidity_range_synergy +
        0.2 * volatility_weighted_momentum +
        0.15 * regime_shift_signal +
        0.05 * intraday_momentum +
        0.05 * efficiency_liquidity_alpha
    )
    
    # Clean up intermediate columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'daily_efficiency', 
             'efficiency_5ma', 'efficiency_slope', 'volume_momentum', 'large_trade_ratio',
             'range_expansion', 'efficiency_regime', 'atr_5', 'volatility_ratio',
             'volatility_slope', 'short_momentum', 'medium_momentum', 'momentum_convergence',
             'morning_strength', 'afternoon_support', 'morning_efficiency', 'afternoon_efficiency'], 
            axis=1, inplace=True, errors='ignore')
    
    return alpha
