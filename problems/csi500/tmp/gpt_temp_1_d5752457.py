import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate a novel alpha factor combining volume-weighted momentum with volatility smoothing,
    liquidity regime adaptation, order flow analysis, range breakout detection, and multi-timeframe reversal signals.
    """
    df = data.copy()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Volume-Weighted Price Momentum Factor
    # Calculate Price Momentum Components
    df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['momentum_60'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
    
    # Compute Volume Weighting Scheme
    df['volume_trend'] = df['volume'] / df['volume'].shift(1)
    
    # Volume persistence: Count consecutive days volume > 20-day average
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    df['volume_above_ma'] = df['volume'] > volume_ma_20
    df['volume_persistence'] = df['volume_above_ma'].rolling(window=20).apply(lambda x: x[::-1].cumprod()[::-1].sum(), raw=False)
    
    df['volume_momentum'] = (df['volume'] - df['volume'].shift(5)) / df['volume'].shift(5)
    
    # Combine with Volatility Smoothing
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(window=20).std()
    
    # Apply exponential weighting to momentum terms
    alpha = 0.1
    df['weighted_momentum_5'] = df['momentum_5'].ewm(alpha=alpha).mean()
    df['weighted_momentum_20'] = df['momentum_20'].ewm(alpha=alpha).mean()
    df['weighted_momentum_60'] = df['momentum_60'].ewm(alpha=alpha).mean()
    
    # Volume-weighted momentum factor
    df['vw_momentum'] = (df['weighted_momentum_5'] * df['volume_persistence'] + 
                         df['weighted_momentum_20'] * df['volume_persistence'] + 
                         df['weighted_momentum_60'] * df['volume_persistence']) / df['volatility_20']
    
    # Liquidity Regime Adaptive Factor
    # Assess Market Liquidity Conditions
    df['effective_spread'] = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['price_impact'] = abs(df['close'] - df['open']) / ((df['high'] + df['low']) / 2)
    
    # Detect Regime Transitions
    spread_change = df['effective_spread'].pct_change()
    df['liquidity_shock'] = ((abs(spread_change) > spread_change.rolling(window=20).std() * 2) & 
                            (df['volume_concentration'] > 1.5)).astype(int)
    
    # Regime persistence and strength
    df['regime_persistence'] = df['liquidity_shock'].rolling(window=20).sum()
    df['liquidity_deviation'] = (df['effective_spread'] - df['effective_spread'].rolling(window=20).mean()) / df['effective_spread'].rolling(window=20).std()
    
    # Generate Adaptive Signals
    high_liquidity_mask = df['effective_spread'] < df['effective_spread'].rolling(window=20).quantile(0.3)
    low_liquidity_mask = df['effective_spread'] > df['effective_spread'].rolling(window=20).quantile(0.7)
    
    df['momentum_enhanced'] = df['momentum_20'] * 1.5
    df['reversal_signal'] = -df['momentum_5']
    
    df['adaptive_signal'] = np.where(high_liquidity_mask, df['momentum_enhanced'],
                                   np.where(low_liquidity_mask, df['reversal_signal'], df['momentum_20']))
    
    # Order Flow Imbalance Factor
    # Analyze Amount-Based Flow
    df['directional_amount'] = df['amount'] * np.sign(df['close'] - df['open'])
    df['net_flow_5d'] = df['directional_amount'].rolling(window=5).sum()
    df['total_amount_5d'] = df['amount'].rolling(window=5).sum()
    df['flow_intensity'] = abs(df['net_flow_5d']) / df['total_amount_5d']
    
    # Assess Flow Persistence
    df['flow_direction'] = np.sign(df['directional_amount'])
    df['consecutive_flow_days'] = (df['flow_direction'] == df['flow_direction'].shift(1)).astype(int)
    df['consecutive_flow_count'] = df['consecutive_flow_days'].rolling(window=10).apply(lambda x: x[::-1].cumprod()[::-1].sum(), raw=False)
    
    df['flow_acceleration'] = df['net_flow_5d'].diff()
    df['flow_exhaustion'] = df['flow_intensity'] * df['consecutive_flow_count']
    
    # Generate Predictive Signals
    df['early_momentum_signal'] = np.where((df['consecutive_flow_count'] >= 3) & (df['flow_intensity'] > 0.6), 
                                          df['net_flow_5d'], 0)
    df['reversal_imminent'] = np.where((df['flow_exhaustion'] > df['flow_exhaustion'].rolling(window=20).quantile(0.8)) & 
                                     (df['consecutive_flow_count'] >= 5), -df['net_flow_5d'], 0)
    
    df['order_flow_signal'] = df['early_momentum_signal'] + df['reversal_imminent']
    
    # Volatility-Smoothed Range Breakout
    # Identify Breakout Patterns
    df['normalized_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['range_ma_10'] = df['normalized_range'].rolling(window=10).mean()
    df['range_expansion'] = df['normalized_range'] / df['range_ma_10']
    df['breakout_strength'] = abs(df['close'] - df['open']) / df['normalized_range']
    
    # Apply Volatility Smoothing
    df['volatility_adjusted_range'] = df['normalized_range'] / df['volatility_20']
    df['breakout_confidence'] = df['volume'] * df['range_expansion']
    df['smooth_breakout'] = df['breakout_confidence'].ewm(alpha=0.15).mean()
    
    # Generate Breakout Signals
    high_confidence_mask = (df['smooth_breakout'] > df['smooth_breakout'].rolling(window=20).quantile(0.7)) & (df['range_expansion'] > 1.2)
    false_breakout_mask = (df['range_expansion'] > 1.5) & (df['volume_concentration'] < 0.8)
    
    df['breakout_signal'] = np.where(high_confidence_mask, df['breakout_strength'],
                                   np.where(false_breakout_mask, -df['breakout_strength'], 0))
    
    # Multi-Timeframe Price Reversal
    # Calculate Price Deviation Metrics
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_60'] = df['close'].rolling(window=60).mean()
    
    df['deviation_5'] = (df['close'] - df['ma_5']) / df['ma_5']
    df['deviation_20'] = (df['close'] - df['ma_20']) / df['ma_20']
    df['deviation_60'] = (df['close'] - df['ma_60']) / df['ma_60']
    
    # Assess Reversal Conditions
    df['volume_spike'] = df['volume'] > df['volume'].rolling(window=20).quantile(0.8)
    
    # Generate Reversal Signals
    immediate_rev_mask = abs(df['deviation_5']) > df['deviation_5'].rolling(window=20).std() * 2
    gradual_rev_mask = abs(df['deviation_20']) > df['deviation_20'].rolling(window=20).std() * 1.5
    trend_exhaustion_mask = (abs(df['deviation_60']) > df['deviation_60'].rolling(window=20).std() * 1.2) & df['volume_spike']
    
    df['immediate_reversal'] = np.where(immediate_rev_mask, -df['deviation_5'], 0)
    df['gradual_reversal'] = np.where(gradual_rev_mask, -df['deviation_20'], 0)
    df['trend_exhaustion'] = np.where(trend_exhaustion_mask, -df['deviation_60'], 0)
    
    df['reversal_signal_combined'] = df['immediate_reversal'] + df['gradual_reversal'] + df['trend_exhaustion']
    
    # Combine all factors with equal weighting
    factor_components = ['vw_momentum', 'adaptive_signal', 'order_flow_signal', 'breakout_signal', 'reversal_signal_combined']
    
    # Normalize each component
    for component in factor_components:
        df[f'{component}_norm'] = (df[component] - df[component].rolling(window=60).mean()) / df[component].rolling(window=60).std()
    
    # Final factor combination
    factor = (df['vw_momentum_norm'] * 0.25 + 
              df['adaptive_signal_norm'] * 0.20 + 
              df['order_flow_signal_norm'] * 0.20 + 
              df['breakout_signal_norm'] * 0.15 + 
              df['reversal_signal_combined_norm'] * 0.20)
    
    return factor
