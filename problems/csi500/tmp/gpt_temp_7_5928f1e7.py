import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Volatility Scaling with Price-Volume Divergence alpha factor
    """
    # Calculate daily returns
    daily_returns = df['close'] / df['close'].shift(1) - 1
    
    # Volatility Regime Detection
    short_term_vol = daily_returns.rolling(window=5).std()
    medium_term_vol = daily_returns.rolling(window=20).std()
    
    # Volatility Regime Classification
    regime_indicator = (short_term_vol > medium_term_vol).astype(int)
    
    # Regime-Adaptive Momentum
    short_term_momentum = df['close'] / df['close'].shift(5) - 1
    medium_term_momentum = df['close'] / df['close'].shift(20) - 1
    
    # Regime-Based Momentum Selection
    regime_selected_momentum = np.where(
        regime_indicator == 1,  # High volatility regime
        short_term_momentum,
        medium_term_momentum    # Low volatility regime
    )
    
    # Volume Divergence Analysis
    daily_volume_changes = df['volume'] / df['volume'].shift(1) - 1
    
    # Rolling Price-Volume Correlation (10-day)
    price_volume_corr = pd.Series(index=df.index)
    for i in range(10, len(df)):
        window_returns = daily_returns.iloc[i-9:i+1]
        window_volume_changes = daily_volume_changes.iloc[i-9:i+1]
        valid_mask = (~window_returns.isna()) & (~window_volume_changes.isna())
        if valid_mask.sum() >= 5:  # Minimum 5 valid observations
            price_volume_corr.iloc[i] = np.corrcoef(
                window_returns[valid_mask], 
                window_volume_changes[valid_mask]
            )[0, 1]
        else:
            price_volume_corr.iloc[i] = 0
    
    # Volume Trend Strength
    volume_trend_strength = df['volume'] / df['volume'].shift(5) - 1
    
    # Volume Divergence Adjustment
    volume_adjustment = np.where(
        price_volume_corr > 0,  # Positive correlation: volume confirms price
        volume_trend_strength,
        -volume_trend_strength   # Negative correlation: volume diverges from price
    )
    
    # Factor Integration
    final_alpha = regime_selected_momentum * volume_adjustment
    
    return pd.Series(final_alpha, index=df.index)
