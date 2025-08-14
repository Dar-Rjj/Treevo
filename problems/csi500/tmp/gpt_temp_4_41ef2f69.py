import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Parameters
    momentum_lookback = 20
    volume_lookback = 20
    trend_vol_lookback = 14
    spike_vol_lookback = 5
    
    # Calculate Momentum Score
    df['close_change'] = df['close'] - df['close'].shift(momentum_lookback)
    df['momentum_score'] = df['close_change'].rolling(window=momentum_lookback).sum()
    
    # Adjust for Volume Volatility
    df['avg_volume'] = df['volume'].rolling(window=volume_lookback).mean()
    df['volume_deviation'] = df['volume'] - df['avg_volume']
    df['volume_adjustment_factor'] = (df['volume_deviation'] + 1e-6) / df['avg_volume']
    df['adjusted_momentum'] = df['momentum_score'] / df['volume_adjustment_factor']
    
    # Calculate Daily Price Movement Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Determine Daily Return Deviation from VWAP
    df['return_deviation'] = df['close'] - df['vwap']
    
    # Identify Trend Reversal Potential
    df['prev_return_deviation'] = df['return_deviation'].shift(1)
    df['reversal_potential'] = (df['return_deviation'] > df['prev_return_deviation']) & (df['volume'] > df['volume'].rolling(window=trend_vol_lookback).mean())
    
    # Enhance Trend Analysis
    df['price_range_overlook'] = (df['high'] - df['low']).rolling(window=momentum_lookback).sum() / df['price_range'].mean()
    df['volume_change'] = df['volume'] - df['volume'].shift(momentum_lookback)
    df['volume_spike'] = df['volume'] > df['volume'].rolling(window=spike_vol_lookback).mean()
    
    # Final Score Calculation
    df['final_score'] = (
        (df['reversal_potential'] * 1.5) +  # Stronger indication of reversal
        (df['price_range_overlook'] * 1.2 if df['volume_spike'] else 0)  # Additional weight if significant
    )
    
    return df['final_score'].fillna(0)
