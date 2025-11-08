import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate composite alpha factor combining multiple persistence-based signals
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    alpha_signal = pd.Series(index=data.index, dtype=float)
    
    # 1. Intraday Momentum Persistence Factor
    # Calculate intraday momentum efficiency
    intraday_return = (data['close'] - data['open']) / data['open']
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    efficiency_ratio = intraday_return / daily_range.replace(0, np.nan)
    
    # Calculate momentum persistence
    momentum_direction = np.sign(intraday_return)
    persistence_count = pd.Series(index=data.index, dtype=float)
    cumulative_momentum = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if momentum_direction.iloc[i] == momentum_direction.iloc[i-1]:
            persistence_count.iloc[i] = persistence_count.iloc[i-1] + 1
            cumulative_momentum.iloc[i] = cumulative_momentum.iloc[i-1] + intraday_return.iloc[i]
        else:
            persistence_count.iloc[i] = 1
            cumulative_momentum.iloc[i] = intraday_return.iloc[i]
    
    # Apply exponential decay
    lambda_val = 0.9
    decayed_persistence = persistence_count * (lambda_val ** persistence_count)
    
    # Volume confirmation
    volume_ratio = data['volume'] / data['volume'].shift(1)
    
    # Intraday momentum signal
    momentum_signal = decayed_persistence * efficiency_ratio * volume_ratio
    
    # 2. Volatility Regime Persistence Detector
    # Multi-timeframe volatility assessment
    daily_range_pct = (data['high'] - data['low']) / data['close'].shift(1)
    short_term_vol = daily_range_pct.rolling(window=5).std()
    medium_term_vol = daily_range_pct.rolling(window=20).std()
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Volatility regime persistence
    regime_threshold = 1.0
    regime_direction = (volatility_ratio > regime_threshold).astype(int)
    regime_persistence = pd.Series(index=data.index, dtype=float)
    regime_strength = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if regime_direction.iloc[i] == regime_direction.iloc[i-1]:
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] + 1
            regime_strength.iloc[i] = regime_strength.iloc[i-1] + volatility_ratio.iloc[i]
        else:
            regime_persistence.iloc[i] = 1
            regime_strength.iloc[i] = volatility_ratio.iloc[i]
    
    # Volume persistence
    volume_ema_10 = data['volume'].ewm(span=10).mean()
    volume_persistence_ratio = data['volume'] / volume_ema_10
    
    # Volatility regime signal
    regime_signal = regime_persistence * regime_strength * volume_persistence_ratio
    
    # 3. Composite Liquidity Mean Reversion
    # Multi-dimensional liquidity assessment
    price_efficiency = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    volume_concentration = data['volume'] / data['volume'].rolling(window=5).sum()
    amount_efficiency = data['amount'] / (data['high'] - data['low']).replace(0, np.nan)
    
    # Liquidity composite (weighted combination)
    liquidity_composite = (0.4 * price_efficiency + 
                          0.3 * volume_concentration + 
                          0.3 * amount_efficiency)
    
    # Liquidity persistence
    liquidity_threshold = liquidity_composite.rolling(window=20).mean()
    liquidity_direction = (liquidity_composite > liquidity_threshold).astype(int)
    liquidity_persistence = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if liquidity_direction.iloc[i] == liquidity_direction.iloc[i-1]:
            liquidity_persistence.iloc[i] = liquidity_persistence.iloc[i-1] + 1
        else:
            liquidity_persistence.iloc[i] = 1
    
    # Adaptive reversal signal
    price_ema_5 = data['close'].ewm(span=5).mean()
    price_deviation = data['close'] / price_ema_5 - 1
    
    # Liquidity mean reversion signal
    liquidity_signal = liquidity_persistence * price_deviation * liquidity_composite
    
    # 4. Order Flow Persistence Factor
    # Directional flow analysis
    directional_amount = np.sign(data['close'] - data['close'].shift(1)) * data['amount']
    flow_direction = np.sign(directional_amount)
    flow_persistence = pd.Series(index=data.index, dtype=float)
    cumulative_flow = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if flow_direction.iloc[i] == flow_direction.iloc[i-1]:
            flow_persistence.iloc[i] = flow_persistence.iloc[i-1] + 1
            cumulative_flow.iloc[i] = cumulative_flow.iloc[i-1] + directional_amount.iloc[i]
        else:
            flow_persistence.iloc[i] = 1
            cumulative_flow.iloc[i] = directional_amount.iloc[i]
    
    # Flow persistence measurement
    streak_avg_flow = cumulative_flow / flow_persistence.replace(0, np.nan)
    flow_persistence_ratio = directional_amount / streak_avg_flow.replace(0, np.nan)
    
    # Volume confirmation
    streak_avg_volume = data['volume'].rolling(window=5).mean()
    volume_confirmation = data['volume'] / streak_avg_volume
    
    # Order flow signal
    flow_signal = flow_persistence * flow_persistence_ratio * volume_confirmation
    
    # 5. Breakout Persistence Validator
    # Range expansion analysis
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    tr_ema_10 = true_range.ewm(span=10).mean()
    range_expansion = true_range / tr_ema_10
    
    # Expansion persistence
    expansion_threshold = 1.0
    expansion_direction = (range_expansion > expansion_threshold).astype(int)
    expansion_persistence = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if expansion_direction.iloc[i] == expansion_direction.iloc[i-1]:
            expansion_persistence.iloc[i] = expansion_persistence.iloc[i-1] + 1
        else:
            expansion_persistence.iloc[i] = 1
    
    # Volume persistence confirmation
    volume_above_avg = (data['volume'] > volume_ema_10).astype(int)
    volume_persistence_count = pd.Series(index=data.index, dtype=float)
    
    for i in range(1, len(data)):
        if volume_above_avg.iloc[i] == volume_above_avg.iloc[i-1]:
            volume_persistence_count.iloc[i] = volume_persistence_count.iloc[i-1] + 1
        else:
            volume_persistence_count.iloc[i] = 1
    
    # Breakout confidence
    breakout_confidence = expansion_persistence * volume_persistence_count * range_expansion
    
    # Breakout signal
    breakout_signal = breakout_confidence * volume_persistence_ratio
    
    # Composite alpha signal (equal weighted combination)
    signals = pd.DataFrame({
        'momentum': momentum_signal,
        'regime': regime_signal,
        'liquidity': liquidity_signal,
        'flow': flow_signal,
        'breakout': breakout_signal
    })
    
    # Normalize each signal component
    normalized_signals = signals.apply(lambda x: (x - x.mean()) / x.std())
    
    # Generate final composite alpha
    alpha_signal = normalized_signals.mean(axis=1)
    
    return alpha_signal
