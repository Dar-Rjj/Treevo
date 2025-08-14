import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Return Deviation from Close
    df['daily_return'] = df['close'].pct_change()
    
    # Compare Daily Return Deviation with VWAP
    df['vwap_deviation'] = df['close'] - df['amount'] / df['volume']
    df['trend_reversal'] = (df['vwap_deviation'] > df['vwap_deviation'].shift(1)).astype(int)
    
    # Check for Volume Increase
    df['volume_14_ma'] = df['volume'].rolling(window=14).mean()
    df['volume_increase'] = (df['volume'] > df['volume_14_ma']).astype(int)
    
    # Compute Intraday Return
    df['intraday_return'] = (df['high'] - df['open']) / df['open']
    
    # Adjust Intraday Return for Volume
    df['volume_7_ma'] = df['volume'].rolling(window=7).mean()
    df['volume_diff'] = df['volume'] - df['volume_7_ma']
    df['adjusted_intraday_return'] = df['intraday_return'] * df['volume_diff']
    
    # Incorporate Price Volatility
    df['price_volatility'] = df['close'].rolling(window=7).std()
    df['volatility_factor'] = 2.0 if df['price_volatility'] > df['price_volatility'].mean() else 0.8
    df['adjusted_intraday_return'] = df['adjusted_intraday_return'] * df['volatility_factor']
    
    # Compute Price Momentum Factor
    n = 5
    df['price_momentum'] = df['daily_return'].rolling(window=n).sum()
    
    # Combine Price and Volume Momentum
    k1 = 1.0
    k2 = 0.5
    df['combined_momentum'] = (k1 * df['price_momentum']) + (k2 * df['volume_increase'])
    
    # Adjust Combined Momentum for Volume Volatility
    df['volume_20_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_deviation'] = df['volume'] - df['volume_20_ma']
    df['volume_adjustment_factor'] = df['volume_deviation'] + 0.01
    df['adjusted_combined_momentum'] = df['combined_momentum'] / df['volume_adjustment_factor']
    
    # Incorporate Momentum Shift
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['momentum_shift'] = 0
    df.loc[df['sma_5'] > df['sma_20'], 'momentum_shift'] = 1.0
    df.loc[df['sma_5'] < df['sma_20'], 'momentum_shift'] = -1.0
    
    # Introduce Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_indicator'] = 0
    df.loc[df['rsi'] < 30, 'rsi_indicator'] = 1.0
    df.loc[df['rsi'] > 70, 'rsi_indicator'] = -1.0
    
    # Combine Indicators
    df['composite_factor'] = (
        2.0 * df['trend_reversal'] * df['volume_increase'] +
        df['adjusted_intraday_return'] +
        df['adjusted_combined_momentum'] +
        df['momentum_shift'] +
        df['rsi_indicator']
    )
    
    return df['composite_factor']
