import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Multi-Timeframe Momentum & Breakout Efficiency factor
    """
    # Multi-Timeframe Volatility Regime Classification
    df = df.copy()
    
    # Short-term volatility: 5-day High-Low range mean
    short_term_vol = (df['high'] - df['low']).rolling(window=5).mean()
    
    # Medium-term volatility: 10-day return standard deviation
    returns = df['close'].pct_change()
    medium_term_vol = returns.rolling(window=10).std()
    
    # Long-term volatility: 20-day Open-Close spread mean
    long_term_vol = (df['close'] - df['open']).abs().rolling(window=20).mean()
    
    # Volatility regime classification
    vol_regime = pd.Series(index=df.index, dtype='float64')
    vol_combined = (short_term_vol.rank(pct=True) + 
                   medium_term_vol.rank(pct=True) + 
                   long_term_vol.rank(pct=True)) / 3
    vol_regime = np.where(vol_combined > 0.7, 2,  # High volatility
                 np.where(vol_combined < 0.3, 0,  # Low volatility
                         1))  # Medium volatility
    
    # Multi-Timeframe Momentum Analysis
    mom_5d = df['close'].pct_change(5)
    mom_10d = df['close'].pct_change(10)
    mom_20d = df['close'].pct_change(20)
    
    # Momentum persistence assessment
    mom_persistence = pd.Series(index=df.index, dtype='float64')
    for i in range(5, len(df)):
        recent_returns = returns.iloc[i-4:i+1]
        same_dir_count = 0
        for j in range(1, len(recent_returns)):
            if recent_returns.iloc[j] * recent_returns.iloc[j-1] > 0:
                same_dir_count += 1
        mom_persistence.iloc[i] = same_dir_count / 4
    
    # Multi-timeframe momentum alignment
    mom_alignment = ((mom_5d > 0) & (mom_10d > 0) & (mom_20d > 0)).astype(int) - \
                   ((mom_5d < 0) & (mom_10d < 0) & (mom_20d < 0)).astype(int)
    
    # Volume-Expansion Breakout Detection
    vol_20d_avg = df['volume'].rolling(window=20).mean()
    volume_surge = df['volume'] > (2 * vol_20d_avg)
    
    # Price behavior during volume surges
    intraday_efficiency = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6)
    surge_efficiency = intraday_efficiency * volume_surge
    
    # Breakout confirmation
    breakout_confirmed = pd.Series(False, index=df.index)
    for i in range(1, len(df)):
        if volume_surge.iloc[i-1] and df['close'].iloc[i] > df['high'].iloc[i-1]:
            breakout_confirmed.iloc[i] = True
    
    # Intraday Momentum & Efficiency Patterns
    # Simplified intraday patterns using OHLC data
    morning_strength = (df['high'] - df['open']) / (df['high'] - df['low']).replace(0, 1e-6)
    afternoon_weakness = (df['close'] - df['high']) / (df['high'] - df['low']).replace(0, 1e-6)
    
    intraday_divergence = morning_strength * afternoon_weakness
    
    # Support-Resistance Break Detection
    # Short-term: 20-day local maxima/minima
    resistance_20d = df['high'].rolling(window=20).max()
    support_20d = df['low'].rolling(window=20).min()
    
    # Breakout strength measurement
    break_strength = pd.Series(index=df.index, dtype='float64')
    for i in range(len(df)):
        if df['close'].iloc[i] > resistance_20d.iloc[i]:
            break_strength.iloc[i] = (df['close'].iloc[i] - resistance_20d.iloc[i]) / \
                                   (resistance_20d.iloc[i] - support_20d.iloc[i]).replace(0, 1e-6)
        elif df['close'].iloc[i] < support_20d.iloc[i]:
            break_strength.iloc[i] = (df['close'].iloc[i] - support_20d.iloc[i]) / \
                                   (resistance_20d.iloc[i] - support_20d.iloc[i]).replace(0, 1e-6)
        else:
            break_strength.iloc[i] = 0
    
    # Regime-Adaptive Signal Integration
    factor = pd.Series(index=df.index, dtype='float64')
    
    for i in range(20, len(df)):
        # High volatility regime signals
        if vol_regime[i] == 2:
            regime_factor = (
                mom_alignment.iloc[i] * 0.3 +
                surge_efficiency.iloc[i] * 0.4 +
                intraday_divergence.iloc[i] * 0.3
            )
        # Low volatility regime signals
        elif vol_regime[i] == 0:
            regime_factor = (
                mom_persistence.iloc[i] * 0.4 +
                breakout_confirmed.iloc[i] * 0.4 +
                intraday_efficiency.iloc[i] * 0.2
            )
        # Medium volatility regime
        else:
            regime_factor = (
                mom_5d.iloc[i] * 0.2 +
                mom_10d.iloc[i] * 0.2 +
                mom_20d.iloc[i] * 0.2 +
                break_strength.iloc[i] * 0.2 +
                intraday_efficiency.iloc[i] * 0.2
            )
        
        # Multi-horizon integration
        short_term = (mom_5d.iloc[i] + surge_efficiency.iloc[i] + intraday_divergence.iloc[i]) / 3
        medium_term = (mom_10d.iloc[i] + mom_persistence.iloc[i] + break_strength.iloc[i]) / 3
        long_term = (mom_20d.iloc[i] + regime_factor + vol_combined.iloc[i]) / 3
        
        # Final factor combining all horizons
        factor.iloc[i] = (
            short_term * 0.4 +    # 1-3 days
            medium_term * 0.35 +  # 5-10 days  
            long_term * 0.25      # 15-20 days
        )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=50).mean()) / factor.rolling(window=50).std()
    
    return factor.fillna(0)
