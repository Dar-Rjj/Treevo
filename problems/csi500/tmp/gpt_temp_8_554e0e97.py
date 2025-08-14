import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Movement Range
    df['daily_range'] = df['high'] - df['low']
    
    # Determine Daily Return Deviation from Close
    df['daily_return_deviation'] = df['close'] - df['close'].shift(1)
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Compare Daily Return Deviation with VWAP
    df['deviation_from_vwap'] = df['close'] - df['vwap']
    df['trend_reversal_indicator'] = (df['deviation_from_vwap'] > df['deviation_from_vwap'].shift(1)).astype(int)
    
    # Check for Volume Increase
    df['volume_ma_14'] = df['volume'].rolling(window=14).mean()
    df['volume_increase'] = (df['volume'] > df['volume_ma_14']).astype(int)
    
    # Compute Intraday Return
    df['intraday_return'] = (df['high'] - df['open']) / df['open']
    
    # Adjust Intraday Return for Volume
    df['volume_ma_7'] = df['volume'].rolling(window=7).mean()
    df['volume_diff'] = df['volume'] - df['volume_ma_7']
    df['adjusted_intraday_return'] = df['intraday_return'] * df['volume_diff']
    
    # Incorporate Price Volatility
    df['close_std_7'] = df['close'].rolling(window=7).std()
    df['volatility_adjusted_intraday_return'] = df['adjusted_intraday_return'] * (2.5 if df['close_std_7'] > df['close_std_7'].median() else 0.7)
    
    # Compute Price Momentum Factor
    n = 5
    df['price_momentum'] = df['daily_return_deviation'].rolling(window=n).sum()
    
    # Compute Volume Momentum Factor
    m = 5
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_momentum'] = df['volume_change'].rolling(window=m).sum()
    
    # Combine Price and Volume Momentum
    k1 = 1.0
    k2 = 0.5
    df['combined_momentum'] = (k1 * df['price_momentum']) + (k2 * df['volume_momentum'])
    
    # Adjust Combined Momentum for Volume Volatility
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_deviation'] = df['volume'] - df['volume_ma_20']
    df['volume_adjustment_factor'] = df['volume_deviation'] + 0.01  # Small constant to avoid division by zero
    df['adjusted_combined_momentum'] = df['combined_momentum'] / df['volume_adjustment_factor']
    
    # Incorporate Momentum Shift
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['momentum_shift'] = ((df['sma_5'] > df['sma_20']) & (df['sma_5'].shift(1) <= df['sma_20'].shift(1))).astype(int) * 1.5
    df['momentum_shift'] -= ((df['sma_5'] < df['sma_20']) & (df['sma_5'].shift(1) >= df['sma_20'].shift(1))).astype(int) * 1.5
    
    # Introduce Price and Volume Correlation
    correlation_window = 10
    df['price_volume_corr'] = df[['close', 'volume']].rolling(window=correlation_window).corr().unstack().iloc[::2, :].fillna(0)
    df['adjusted_combined_momentum'] *= (1.2 if df['price_volume_corr'] > 0 else 0.8)
    
    # Combine Indicators
    df['trend_reversal_score'] = df['trend_reversal_indicator'] * df['volume_increase'] * 2
    df['final_factor'] = df['trend_reversal_score'] + df['volatility_adjusted_intraday_return'] + df['adjusted_combined_momentum'] + df['momentum_shift']
    
    return df['final_factor']
