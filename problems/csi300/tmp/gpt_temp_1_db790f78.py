import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    df['return_5d'] = df['close'] / df['close'].shift(5) - 1
    df['return_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatility Normalization
    df['daily_range'] = df['high'] - df['low']
    df['range_volatility_10d'] = df['daily_range'].rolling(window=10).mean()
    
    df['norm_momentum_5d'] = df['return_5d'] / df['range_volatility_10d']
    df['norm_momentum_10d'] = df['return_10d'] / df['range_volatility_10d']
    
    # Volume Divergence Detection
    df['volume_momentum_5d'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_trend'] = np.sign(df['volume'] - df['volume'].shift(1))
    
    # Bullish divergence: negative price momentum + positive volume momentum
    bullish_divergence = (df['return_5d'] < 0) & (df['volume_momentum_5d'] > 0)
    # Bearish divergence: positive price momentum + negative volume momentum
    bearish_divergence = (df['return_5d'] > 0) & (df['volume_momentum_5d'] < 0)
    
    # Regime-Based Weighting
    df['range_percentile'] = df['daily_range'].rolling(window=20).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) * 1 + (x.iloc[-1] < np.percentile(x, 20)) * -1, 
        raw=False
    )
    
    # Timeframe selection weights
    df['weight_5d'] = np.where(df['range_percentile'] == -1, 1.0, 
                              np.where(df['range_percentile'] == 1, 0.0, 0.5))
    df['weight_10d'] = np.where(df['range_percentile'] == 1, 1.0, 
                               np.where(df['range_percentile'] == -1, 0.0, 0.5))
    
    # Base Signal: weighted volatility-normalized momentum
    df['base_signal'] = (df['weight_5d'] * df['norm_momentum_5d'] + 
                        df['weight_10d'] * df['norm_momentum_10d'])
    
    # Volume Confirmation
    df['divergence_strength'] = np.abs(df['return_5d']) * np.abs(df['volume_momentum_5d'])
    
    # Apply volume confirmation
    df['volume_confirmation'] = np.where(
        bullish_divergence | bearish_divergence,
        df['divergence_strength'],
        df['volume_trend']
    )
    
    # Final Alpha Factor
    df['alpha_factor'] = df['base_signal'] * df['volume_confirmation']
    
    # Set conflicting signals to zero
    conflicting_signals = (bullish_divergence & (df['base_signal'] > 0)) | \
                         (bearish_divergence & (df['base_signal'] < 0))
    df.loc[conflicting_signals, 'alpha_factor'] = 0
    
    return df['alpha_factor']
