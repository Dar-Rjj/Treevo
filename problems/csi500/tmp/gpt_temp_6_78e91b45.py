import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Novel Alpha Factor combining multiple persistence-based signals with regime adaptation
    """
    df = data.copy()
    
    # Initialize factor series
    factor = pd.Series(index=df.index, dtype=float)
    
    # 1. Momentum Persistence Factor
    # Multi-timeframe momentum
    mom_5 = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    mom_10 = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    mom_20 = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Momentum persistence measurement
    mom_sign_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window_5 = mom_5.iloc[i-4:i+1]
        window_10 = mom_10.iloc[i-9:i+1]
        window_20 = mom_20.iloc[i-19:i+1]
        
        # Count consecutive positive/negative momentum days
        pos_count_5 = (window_5 > 0).sum()
        neg_count_5 = (window_5 < 0).sum()
        persistence_5 = max(pos_count_5, neg_count_5) / 5
        
        pos_count_10 = (window_10 > 0).sum()
        neg_count_10 = (window_10 < 0).sum()
        persistence_10 = max(pos_count_10, neg_count_10) / 10
        
        pos_count_20 = (window_20 > 0).sum()
        neg_count_20 = (window_20 < 0).sum()
        persistence_20 = max(pos_count_20, neg_count_20) / 20
        
        # Exponential smoothing of persistence
        if i == 20:
            mom_persistence = (persistence_5 + persistence_10 + persistence_20) / 3
        else:
            mom_persistence = 0.94 * mom_sign_persistence.iloc[i-1] + 0.06 * ((persistence_5 + persistence_10 + persistence_20) / 3)
        
        # Volume-weighted adjustment
        vol_weight = df['volume'].iloc[i] / df['volume'].rolling(20).mean().iloc[i]
        mom_sign_persistence.iloc[i] = mom_persistence * np.sqrt(vol_weight)
    
    # 2. Volatility-Clustered Reversal
    # True Range calculation
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # EMA of True Range
    tr_ema = true_range.ewm(alpha=0.2).mean()  # λ=0.8 equivalent
    
    # Volatility clustering detection
    vol_cluster = pd.Series(index=df.index, dtype=float)
    high_vol_threshold = tr_ema.rolling(20).quantile(0.7)
    
    for i in range(5, len(df)):
        # Volatility cluster identification
        recent_tr = true_range.iloc[i-4:i+1]
        vol_ratio = recent_tr / tr_ema.iloc[i]
        vol_cluster_count = (vol_ratio > 1.2).sum()
        
        # Price reversal measurement
        daily_ret = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
        cum_ret_3 = (df['close'].iloc[i] - df['close'].iloc[i-3]) / df['close'].iloc[i-3]
        
        # Reversal strength
        if abs(daily_ret) > 0.02 and np.sign(daily_ret) != np.sign(cum_ret_3):
            reversal_strength = abs(daily_ret) * (1 - abs(cum_ret_3))
        else:
            reversal_strength = 0
        
        # Volume confirmation
        vol_persistence = df['volume'].iloc[i] / df['volume'].ewm(span=5).mean().iloc[i]
        vol_cluster.iloc[i] = reversal_strength * vol_persistence * (vol_cluster_count / 5)
    
    # Exponential smoothing of reversal confidence
    reversal_signal = vol_cluster.ewm(alpha=0.1).mean()
    
    # 3. Liquidity-Regime Momentum
    # Multi-dimensional liquidity assessment
    price_efficiency = (df['high'] - df['low']) / df['close'].shift(1)
    volume_intensity = df['volume'] / df['volume'].rolling(20).median()
    amount_density = df['amount'] / (df['high'] - df['low']).replace(0, 0.001)
    
    # Liquidity regime classification
    high_liq_threshold = price_efficiency.rolling(20).median()
    high_vol_intensity = volume_intensity.rolling(20).median()
    high_amount_density = amount_density.rolling(20).median()
    
    liquidity_regime = pd.Series(index=df.index, dtype=int)
    for i in range(20, len(df)):
        if (price_efficiency.iloc[i] > high_liq_threshold.iloc[i] and 
            volume_intensity.iloc[i] > high_vol_intensity.iloc[i] and 
            amount_density.iloc[i] > high_amount_density.iloc[i]):
            liquidity_regime.iloc[i] = 2  # High liquidity
        elif (price_efficiency.iloc[i] < high_liq_threshold.iloc[i] and 
              volume_intensity.iloc[i] < high_vol_intensity.iloc[i] and 
              amount_density.iloc[i] < high_amount_density.iloc[i]):
            liquidity_regime.iloc[i] = 0  # Low liquidity
        else:
            liquidity_regime.iloc[i] = 1  # Mixed liquidity
    
    # Regime-specific momentum
    regime_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        current_mom = mom_5.iloc[i]
        regime = liquidity_regime.iloc[i]
        
        if regime == 2:  # High liquidity - accelerated
            regime_momentum.iloc[i] = current_mom * 1.5
        elif regime == 0:  # Low liquidity - dampened
            regime_momentum.iloc[i] = current_mom * 0.5
        else:  # Mixed liquidity - normal
            regime_momentum.iloc[i] = current_mom
    
    # 4. Order Flow Persistence Alpha
    # Directional flow analysis
    net_amount_flow = np.sign(df['close'] - df['close'].shift(1)) * df['amount']
    cum_net_flow_5 = net_amount_flow.rolling(5).sum()
    
    # Flow intensity measurement
    flow_intensity = abs(net_amount_flow) / df['amount'].rolling(20).mean()
    intensity_persistence = flow_intensity.rolling(5).apply(lambda x: (x > x.median()).sum() / 5)
    
    # Flow direction persistence
    flow_direction = np.sign(net_amount_flow)
    direction_persistence = flow_direction.rolling(5).apply(lambda x: (x == x.iloc[-1]).sum() / 5)
    
    # Predictive signal generation
    flow_signal = direction_persistence * intensity_persistence * np.sign(cum_net_flow_5)
    flow_signal_smoothed = flow_signal.ewm(alpha=0.1).mean()
    
    # 5. Breakout Persistence Factor
    # Range expansion analysis
    range_expansion_ratio = true_range / true_range.ewm(span=10).mean()
    
    # Breakout confirmation
    price_breakout = pd.Series(index=df.index, dtype=float)
    volume_confirmation = df['volume'] / df['volume'].ewm(span=20).mean()
    
    for i in range(10, len(df)):
        # Price level breakthrough detection
        recent_high = df['high'].iloc[i-9:i+1].max()
        recent_low = df['low'].iloc[i-9:i+1].min()
        
        if df['close'].iloc[i] > recent_high:
            price_break = (df['close'].iloc[i] - recent_high) / recent_high
        elif df['close'].iloc[i] < recent_low:
            price_break = (df['close'].iloc[i] - recent_low) / recent_low
        else:
            price_break = 0
        
        # Breakout persistence
        expansion_days = (range_expansion_ratio.iloc[i-4:i+1] > 1.1).sum()
        breakout_persistence = expansion_days / 5
        
        price_breakout.iloc[i] = price_break * breakout_persistence * volume_confirmation.iloc[i]
    
    breakout_confidence = price_breakout.ewm(alpha=0.1).mean()
    
    # Combine all factors with regime adaptation
    volatility_regime = true_range.rolling(20).std() / true_range.rolling(60).std()
    
    for i in range(60, len(df)):
        # Volatility regime adaptation
        if volatility_regime.iloc[i] > 1.2:  # High volatility regime
            persistence_weight = 0.3
            reversal_weight = 0.4
            regime_mom_weight = 0.1
            flow_weight = 0.1
            breakout_weight = 0.1
        else:  # Low volatility regime
            persistence_weight = 0.4
            reversal_weight = 0.2
            regime_mom_weight = 0.2
            flow_weight = 0.1
            breakout_weight = 0.1
        
        # Combine signals
        combined_signal = (
            persistence_weight * mom_sign_persistence.iloc[i] * np.sign(mom_5.iloc[i]) +
            reversal_weight * reversal_signal.iloc[i] +
            regime_mom_weight * regime_momentum.iloc[i] +
            flow_weight * flow_signal_smoothed.iloc[i] +
            breakout_weight * breakout_confidence.iloc[i]
        )
        
        factor.iloc[i] = combined_signal
    
    # Normalize the factor
    factor = (factor - factor.rolling(60).mean()) / factor.rolling(60).std()
    
    return factor
