import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum-Adjusted Volatility Factor
    # Calculate Short-Term Momentum
    momentum_5d = data['close'].pct_change(periods=5)
    momentum_10d = data['close'].pct_change(periods=10)
    short_term_momentum = (momentum_5d + momentum_10d) / 2
    
    # Calculate Rolling Volatility
    high_low_range = (data['high'] - data['low']) / data['close']
    volatility_20d = high_low_range.rolling(window=20).std()
    
    # Combine Momentum and Volatility
    momentum_vol_factor = short_term_momentum * volatility_20d
    
    # Price-Volume Convergence Factor
    # Calculate Trend Components
    price_ema_10d = data['close'].ewm(span=10).mean()
    volume_ema_10d = data['volume'].ewm(span=10).mean()
    
    # Detect Convergence Patterns
    price_trend = price_ema_10d.diff()
    volume_trend = volume_ema_10d.diff()
    
    # Quantify convergence strength
    convergence_strength = pd.Series(
        [price_trend.iloc[i-15:i].corr(volume_trend.iloc[i-15:i]) 
         if i >= 15 else np.nan 
         for i in range(len(price_trend))], 
        index=price_trend.index
    )
    
    price_volume_factor = convergence_strength * np.sign(price_trend * volume_trend)
    
    # Liquidity-Weighted Reversal
    # Identify Extreme Price Movements
    returns_3d = data['close'].pct_change(periods=3)
    
    # Apply Liquidity Adjustment
    liquidity_proxy = data['volume']
    
    # Weight reversal signal
    reversal_factor = -returns_3d / (liquidity_proxy / liquidity_proxy.rolling(window=20).mean())
    
    # Intraday Pattern Consistency
    # Calculate Intraday Metrics
    high_low_efficiency = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    open_close_relationship = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Track Pattern Persistence
    pattern_strength = pd.Series(
        [(high_low_efficiency.iloc[i-5:i] > 1).sum() + (abs(open_close_relationship.iloc[i-5:i]) > 0.5).sum()
         if i >= 5 else np.nan 
         for i in range(len(high_low_efficiency))],
        index=high_low_efficiency.index
    )
    
    intraday_factor = pattern_strength * open_close_relationship
    
    # Volatility Transition Factor
    # Detect Regime Changes
    returns_daily = data['close'].pct_change()
    hist_vol_30d = returns_daily.rolling(window=30).std()
    
    # Identify Transition Signals
    vol_regime_boundary = hist_vol_30d.rolling(window=60).mean()
    regime_crossing = (hist_vol_30d > vol_regime_boundary).astype(int).diff()
    
    # Combine with price direction
    price_momentum = data['close'].pct_change(periods=5)
    volatility_transition_factor = regime_crossing * price_momentum
    
    # Combine all factors with equal weights
    combined_factor = (
        momentum_vol_factor.fillna(0) +
        price_volume_factor.fillna(0) +
        reversal_factor.fillna(0) +
        intraday_factor.fillna(0) +
        volatility_transition_factor.fillna(0)
    ) / 5
    
    return combined_factor
