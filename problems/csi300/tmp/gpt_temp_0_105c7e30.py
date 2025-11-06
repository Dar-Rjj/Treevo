import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price-volume divergence, range efficiency, 
    volume-confirmed reversal, amount flow, and volatility-volume regime signals.
    """
    # Price-Volume Momentum Divergence
    # Price momentum
    ret_5d = df['close'] / df['close'].shift(5) - 1
    ret_10d = df['close'] / df['close'].shift(10) - 1
    
    # Volume momentum
    vol_ratio_5d = df['volume'] / df['volume'].shift(5)
    vol_accel = vol_ratio_5d / (df['volume'].shift(5) / df['volume'].shift(10))
    
    # Divergence signal (bullish: high price momentum + low volume momentum)
    price_momentum = (ret_5d + ret_10d) / 2
    volume_momentum = (vol_ratio_5d + vol_accel) / 2
    divergence_signal = price_momentum.rank(pct=True) - volume_momentum.rank(pct=True)
    
    # Range Efficiency Ratio
    # Daily range
    true_range = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)
    norm_range = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Efficiency metrics
    daily_eff = abs(df['close'] - df['close'].shift(1)) / (df['high'] - df['low'])
    
    # 3-day efficiency
    range_sum_3d = (df['high'] - df['low']) + (df['high'].shift(1) - df['low'].shift(1)) + (df['high'].shift(2) - df['low'].shift(2))
    eff_3d = abs(df['close'] - df['close'].shift(3)) / range_sum_3d
    
    # Efficiency momentum
    eff_trend = eff_3d / daily_eff
    eff_persistence = pd.Series([(daily_eff.iloc[i-5:i] > 0.7).sum() 
                                for i in range(len(daily_eff))], index=daily_eff.index)
    
    efficiency_signal = (eff_trend.rank(pct=True) + eff_persistence.rank(pct=True)) / 2
    
    # Volume-Confirmed Reversal
    # Extreme move detection
    price_dev_3d = (df['close'] - df['close'].shift(3)) / (
        df['high'].rolling(4).max() - df['low'].rolling(4).min())
    vol_spike = df['volume'] / df['volume'].rolling(6).mean()
    
    # Reversal pattern (oversold bounce: low price + high volume spike)
    price_rank = price_dev_3d.rank(pct=True)
    vol_rank = vol_spike.rank(pct=True)
    reversal_signal = (1 - price_rank) * vol_rank  # Oversold bounce signal
    
    # Amount Flow Direction
    # Flow classification
    up_amount = df['amount'].where(df['close'] > df['close'].shift(1), 0)
    down_amount = df['amount'].where(df['close'] < df['close'].shift(1), 0)
    
    # Flow momentum
    net_flow = up_amount - down_amount
    flow_accel = net_flow / net_flow.shift(3)
    
    # Flow persistence
    flow_sign_consistency = pd.Series([
        (np.sign(net_flow.iloc[i-5:i]) == np.sign(net_flow.iloc[i-1])).sum() 
        for i in range(len(net_flow))], index=net_flow.index)
    
    flow_regime_strength = net_flow.rolling(4).sum() / df['amount'].rolling(4).sum()
    
    flow_signal = (flow_accel.rank(pct=True) + flow_sign_consistency.rank(pct=True) + 
                  flow_regime_strength.rank(pct=True)) / 3
    
    # Volatility-Volume Regime
    # Volatility measurement
    vol_10d = (df['high'].rolling(11).max() - df['low'].rolling(11).min()) / df['close'].shift(10)
    vol_ratio_5d = df['close'].rolling(6).std() / df['close'].shift(5).rolling(6).std()
    
    # Volume patterns
    vol_spike_cluster = pd.Series([
        (df['volume'].iloc[i-5:i] > 1.5 * df['volume'].iloc[i-10:i].mean()).sum() 
        for i in range(len(df))], index=df.index)
    vol_vol_ratio = df['volume'] / (df['high'] - df['low'])
    
    # Regime signals (high volatility + clustered volume = trend continuation)
    vol_regime = vol_10d.rank(pct=True) * vol_spike_cluster.rank(pct=True)
    
    # Combine all signals with weights
    alpha = (
        0.25 * divergence_signal +           # Price-volume divergence
        0.20 * efficiency_signal +           # Range efficiency
        0.25 * reversal_signal +             # Volume-confirmed reversal
        0.15 * flow_signal +                 # Amount flow direction
        0.15 * vol_regime                    # Volatility-volume regime
    )
    
    return alpha
