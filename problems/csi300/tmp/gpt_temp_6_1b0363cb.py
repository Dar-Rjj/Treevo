import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining price-volume momentum divergence, range efficiency persistence,
    extreme move reversal detection, amount flow direction consistency, and volatility-volume regime signals.
    """
    df = data.copy()
    
    # Price Momentum Components
    df['price_momentum_short'] = df['close'] / df['close'].shift(5) - 1
    df['price_momentum_medium'] = df['close'] / df['close'].shift(10) - 1
    df['price_acceleration'] = (df['close'] / df['close'].shift(5)) / (df['close'].shift(5) / df['close'].shift(10)) - 1
    
    # Volume Momentum Components
    df['volume_trend'] = df['volume'] / df['volume'].shift(5)
    df['volume_acceleration'] = (df['volume'] / df['volume'].shift(5)) / (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Volume persistence (count of days with volume > previous day's volume over last 5 days)
    volume_persistence = []
    for i in range(len(df)):
        if i < 5:
            volume_persistence.append(np.nan)
        else:
            count = sum(df['volume'].iloc[j] > df['volume'].iloc[j-1] for j in range(i-4, i+1))
            volume_persistence.append(count)
    df['volume_persistence'] = volume_persistence
    
    # Divergence Signals
    df['bullish_divergence'] = (df['price_acceleration'] > 0) & (df['volume_trend'] < 1)
    df['bearish_divergence'] = (df['price_momentum_short'] < 0) & (df['volume'] > df['volume'].shift(1) * 1.5)
    
    # Daily Efficiency Metrics
    df['range_efficiency'] = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    df['opening_efficiency'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['gap_efficiency'] = abs(df['open'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # Multi-day Persistence
    efficiency_streak = []
    efficiency_momentum = []
    efficiency_volatility = []
    
    for i in range(len(df)):
        if i < 5:
            efficiency_streak.append(np.nan)
            efficiency_momentum.append(np.nan)
            efficiency_volatility.append(np.nan)
        else:
            # Efficiency streak
            streak = 0
            for j in range(i, max(-1, i-10), -1):
                if j >= 0 and df['range_efficiency'].iloc[j] > 0.7:
                    streak += 1
                else:
                    break
            efficiency_streak.append(streak)
            
            # Efficiency momentum (3-day sum)
            eff_momentum = df['range_efficiency'].iloc[max(0, i-2):i+1].sum()
            efficiency_momentum.append(eff_momentum)
            
            # Efficiency volatility (5-day std)
            eff_vol = df['range_efficiency'].iloc[max(0, i-4):i+1].std()
            efficiency_volatility.append(eff_vol)
    
    df['efficiency_streak'] = efficiency_streak
    df['efficiency_momentum'] = efficiency_momentum
    df['efficiency_volatility'] = efficiency_volatility
    
    # Volume Confirmation
    df['high_efficiency_volume'] = np.where(df['range_efficiency'] > 0.8, df['volume'], 0)
    df['low_efficiency_volume'] = np.where(df['range_efficiency'] < 0.3, df['volume'], 0)
    
    # Extreme Identification
    returns = df['close'].pct_change()
    df['price_extreme'] = abs(returns) > (2 * returns.rolling(window=10).std())
    df['volume_extreme'] = df['volume'] > (2 * df['volume'].rolling(window=10).median())
    df['combined_extreme'] = df['price_extreme'] & df['volume_extreme']
    
    # Reversal Patterns
    df['gap_reversal'] = np.sign(df['open'] - df['close'].shift(1)) != np.sign(df['close'] - df['open'])
    df['range_expansion'] = (df['high'] - df['low']) / (df['high'].shift(1) - df['low'].shift(1))
    
    # Amount Flow Direction Classification
    df['bull_flow'] = (df['close'] > df['close'].shift(1)) & (df['amount'] > df['amount'].shift(1))
    df['bear_flow'] = (df['close'] < df['close'].shift(1)) & (df['amount'] > df['amount'].shift(1))
    df['neutral_flow'] = ~(df['bull_flow'] | df['bear_flow'])
    
    # Flow Persistence
    bull_streak = []
    bear_streak = []
    flow_momentum = []
    
    for i in range(len(df)):
        if i < 5:
            bull_streak.append(np.nan)
            bear_streak.append(np.nan)
            flow_momentum.append(np.nan)
        else:
            # Bull streak
            bull_count = 0
            for j in range(i, max(-1, i-10), -1):
                if j >= 0 and df['bull_flow'].iloc[j]:
                    bull_count += 1
                else:
                    break
            bull_streak.append(bull_count)
            
            # Bear streak
            bear_count = 0
            for j in range(i, max(-1, i-10), -1):
                if j >= 0 and df['bear_flow'].iloc[j]:
                    bear_count += 1
                else:
                    break
            bear_streak.append(bear_count)
            
            # Flow momentum (5-day net bull days)
            flow_mom = (df['bull_flow'].iloc[max(0, i-4):i+1].sum() - 
                       df['bear_flow'].iloc[max(0, i-4):i+1].sum())
            flow_momentum.append(flow_mom)
    
    df['bull_streak'] = bull_streak
    df['bear_streak'] = bear_streak
    df['flow_momentum'] = flow_momentum
    
    # Volatility Regimes
    vol_5d = df['close'].rolling(window=5).std()
    vol_10d = df['close'].rolling(window=10).std()
    df['high_volatility'] = vol_5d > (1.5 * vol_10d.shift(5))
    df['low_volatility'] = vol_5d < (0.7 * vol_10d.shift(5))
    df['normal_volatility'] = ~(df['high_volatility'] | df['low_volatility'])
    
    # Volume Regimes
    vol_median = df['volume'].rolling(window=10).median()
    df['high_volume'] = df['volume'] > (1.5 * vol_median)
    df['low_volume'] = df['volume'] < (0.7 * vol_median)
    df['normal_volume'] = ~(df['high_volume'] | df['low_volume'])
    
    # Combine all signals into final alpha factor
    alpha = (
        # Price-volume divergence component
        (df['bullish_divergence'].astype(int) - df['bearish_divergence'].astype(int)) * 0.2 +
        
        # Range efficiency component
        (df['efficiency_streak'].fillna(0) * df['efficiency_momentum'].fillna(0) * 0.15 -
         df['efficiency_volatility'].fillna(0) * 0.1) +
        
        # Extreme reversal component
        (df['combined_extreme'].astype(int) * df['gap_reversal'].astype(int) * 0.25 -
         df['range_expansion'].fillna(1) * 0.05) +
        
        # Amount flow component
        (df['flow_momentum'].fillna(0) * 0.15 +
         df['bull_streak'].fillna(0) * 0.08 -
         df['bear_streak'].fillna(0) * 0.08) +
        
        # Volatility-volume regime component
        ((df['high_volatility'] & df['high_volume']).astype(int) * 0.12 +
         (df['low_volatility'] & df['high_volume']).astype(int) * -0.12 +
         (df['high_volatility'] & df['low_volume']).astype(int) * -0.08 +
         (df['low_volatility'] & df['low_volume']).astype(int) * 0.05)
    )
    
    return alpha
