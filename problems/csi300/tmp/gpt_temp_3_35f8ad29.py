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
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Classification
    df['tr_20d'] = df['tr'].rolling(window=20).mean()
    df['tr_252d_percentile'] = df['tr_20d'].rolling(window=252).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 70)), raw=False
    )
    
    # Calculate returns for different periods
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_60d'] = df['close'].pct_change(60)
    
    # Volatility Regime Momentum
    high_vol_momentum = (
        0.6 * df['ret_5d'] + 
        0.3 * df['ret_20d'] + 
        0.1 * df['ret_60d']
    )
    low_vol_momentum = (
        0.2 * df['ret_5d'] + 
        0.5 * df['ret_20d'] + 
        0.3 * df['ret_60d']
    )
    df['vol_regime_momentum'] = np.where(
        df['tr_252d_percentile'] == 1, 
        high_vol_momentum, 
        low_vol_momentum
    )
    
    # Volume-Confirmed Breakout
    df['high_20d'] = df['high'].rolling(window=20).max()
    df['low_20d'] = df['low'].rolling(window=20).min()
    df['volume_20d_avg'] = df['volume'].rolling(window=20).mean()
    
    # Breakout conditions
    upside_breakout = (df['close'] > df['high_20d'].shift(1))
    downside_breakout = (df['close'] < df['low_20d'].shift(1))
    volume_confirmation = (df['volume'] > 1.5 * df['volume_20d_avg'])
    
    # Breakout magnitude
    upside_magnitude = (df['close'] - df['high_20d'].shift(1)) / df['high_20d'].shift(1)
    downside_magnitude = (df['low_20d'].shift(1) - df['close']) / df['low_20d'].shift(1)
    
    df['breakout_factor'] = 0
    df.loc[upside_breakout & volume_confirmation, 'breakout_factor'] = upside_magnitude
    df.loc[downside_breakout & volume_confirmation, 'breakout_factor'] = -downside_magnitude
    
    # Liquidity-Adjusted Mean Reversion
    df['ma_20d'] = df['close'].rolling(window=20).mean()
    df['price_deviation'] = (df['close'] - df['ma_20d']) / df['ma_20d']
    
    # Liquidity conditions
    high_liquidity = (df['volume'] > 1.2 * df['volume_20d_avg'])
    low_liquidity = (df['volume'] < 0.8 * df['volume_20d_avg'])
    
    # Mean reversion factor with liquidity adjustment
    df['mean_reversion_factor'] = -df['price_deviation']
    df.loc[high_liquidity, 'mean_reversion_factor'] = df['mean_reversion_factor'] * 1.5
    df.loc[low_liquidity, 'mean_reversion_factor'] = df['mean_reversion_factor'] * 0.5
    
    # Combine factors with equal weights
    df['combined_factor'] = (
        df['vol_regime_momentum'] + 
        df['breakout_factor'] + 
        df['mean_reversion_factor']
    ) / 3
    
    # Clean up intermediate columns
    result = df['combined_factor'].copy()
    
    return result
