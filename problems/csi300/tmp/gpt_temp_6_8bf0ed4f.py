import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum-Volume-Volatility Composite with Bounded Transforms and Regime Awareness
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Divergence
    df['ROC_3'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    df['ROC_8'] = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    df['Mom_Divergence'] = np.tanh(df['ROC_3'] * df['ROC_8']) * np.sign(df['ROC_3'] + df['ROC_8'])
    
    # Volume-Price Confirmation
    df['Vol_ROC_5'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    # Calculate 10-day price-volume correlation
    corr_pv = []
    for i in range(len(df)):
        if i >= 9:
            close_window = df['close'].iloc[i-9:i+1]
            volume_window = df['volume'].iloc[i-9:i+1]
            corr = close_window.corr(volume_window)
            corr_pv.append(corr if not np.isnan(corr) else 0)
        else:
            corr_pv.append(0)
    df['Corr_PV'] = corr_pv
    
    df['Vol_Confirm'] = np.tanh(df['Vol_ROC_5']) * (1 + df['Corr_PV'])
    
    # Volatility-Adjusted Momentum
    df['Mom_Vol'] = df['Mom_Divergence'] * df['Vol_Confirm']
    
    # True range calculation
    df['TR'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # Average true range (15-day)
    df['ATR_15'] = df['TR'].rolling(window=15, min_periods=1).mean()
    
    # Volatility-scaled factor
    df['Vol_Scaled'] = df['Mom_Vol'] / df['ATR_15']
    
    # Regime-Aware Final Factor
    df['Volatility_ratio'] = df['TR'] / df['ATR_15']
    df['High_vol_regime'] = (df['Volatility_ratio'] > 1.8).astype(float)
    
    # Market state indicator
    df['Market_trend'] = np.sign(
        df['close'].rolling(window=5).sum() - df['close'].shift(5).rolling(window=5).sum()
    )
    
    # Regime-adjusted alpha
    df['Alpha'] = df['Vol_Scaled'] * (1 - 0.25 * df['High_vol_regime']) * (1 + 0.15 * df['Market_trend'])
    
    return df['Alpha']
