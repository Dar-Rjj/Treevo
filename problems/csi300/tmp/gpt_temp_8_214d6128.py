import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(data):
    """
    Momentum-Volume Divergence Factor with regime-aware scaling
    """
    df = data.copy()
    
    # Calculate Price Momentum
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_5d'] = df['close'].pct_change(5)
    
    # Calculate Volume Trend using linear regression slopes
    def volume_slope(window, period):
        x = np.arange(period)
        slopes = []
        for i in range(len(window) - period + 1):
            y = window.iloc[i:i+period].values
            if len(y) == period and not np.isnan(y).any():
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=window.index[period-1:])
    
    df['volume_slope_10d'] = volume_slope(df['volume'], 10)
    df['volume_slope_5d'] = volume_slope(df['volume'], 5)
    
    # Compute Divergence Score
    df['momentum_volume_divergence'] = (
        df['momentum_5d'] * np.sign(df['momentum_5d']) * 
        (-df['volume_slope_5d']) * np.sign(-df['volume_slope_5d'])
    )
    
    # Scale by momentum strength
    df['momentum_strength'] = np.abs(df['momentum_5d'])
    df['divergence_magnitude'] = df['momentum_volume_divergence'] * df['momentum_strength']
    
    # Calculate True Range and ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            np.abs(df['high'] - df['close'].shift(1)),
            np.abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_20d'] = df['tr'].rolling(window=20).mean()
    df['atr_60d_median'] = df['atr_20d'].rolling(window=60).median()
    
    # Volatility regime scaling
    df['volatility_regime'] = df['atr_20d'] / df['atr_60d_median']
    df['volatility_scaling'] = np.where(
        df['volatility_regime'] > 1.2,  # High volatility threshold
        0.7,  # Scale down in high volatility
        1.0   # Maintain in low volatility
    )
    
    # Calculate market correlation (using close price as market proxy)
    df['stock_returns'] = df['close'].pct_change()
    df['market_returns'] = df['close'].pct_change()  # Using same stock as market proxy for demonstration
    df['correlation_20d'] = df['stock_returns'].rolling(window=20).corr(df['market_returns'])
    
    # Market condition scaling
    df['correlation_scaling'] = np.where(
        np.abs(df['correlation_20d']) > 0.7,  # High correlation threshold
        0.6,  # Reduce weight for high systematic exposure
        1.0   # Maintain weight for idiosyncratic moves
    )
    
    # Final factor calculation
    df['factor'] = (
        df['divergence_magnitude'] * 
        df['volatility_scaling'] * 
        df['correlation_scaling']
    )
    
    # Clean and return
    factor_series = df['factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor_series
