import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate simple moving average (SMA) of close prices
    df['SMA'] = df['close'].rolling(window=20).mean()
    
    # Calculate exponential moving average (EMA) of close prices
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate volume-averaged price (VAP)
    df['VAP'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Calculate simple moving average of VAP
    df['SMA_VAP'] = df['VAP'].rolling(window=20).mean()
    
    # Determine if SMA crosses above or below EMA
    df['SMA_above_EMA'] = (df['SMA'] > df['EMA']).astype(int)
    
    # Count the number of days where SMA > EMA in a window
    df['SMA_above_EMA_count'] = df['SMA_above_EMA'].rolling(window=20).sum()
    
    # Determine if VAP crosses above or below SMA_VAP
    df['VAP_above_SMA_VAP'] = (df['VAP'] > df['SMA_VAP']).astype(int)
    
    # Calculate daily price range (high - low)
    df['daily_price_range'] = df['high'] - df['low']
    
    # Compute average true range (ATR) over a period
    df['TR'] = df[['high', 'close']].shift(1).max(axis=1) - df[['low', 'close']].shift(1).min(axis=1)
    df['ATR'] = df['TR'].rolling(window=20).mean()
    
    # Calculate price range averaged over a period
    df['avg_daily_price_range'] = df['daily_price_range'].rolling(window=20).mean()
    
    # Total volume over a specified period
    df['total_volume'] = df['volume'].rolling(window=20).sum()
    
    # Daily volume change
    df['daily_volume_change'] = df['volume'] - df['volume'].shift(1)
    
    # Total amount traded over a specified period
    df['total_amount'] = df['amount'].rolling(window=20).sum()
    
    # Daily amount change
    df['daily_amount_change'] = df['amount'] - df['amount'].shift(1)
    
    # Calculate volume-to-amount ratio
    df['volume_to_amount_ratio'] = df['volume'] / df['amount']
    
    # Identify days where price and volume move in opposite directions
    df['price_vol_divergence'] = ((df['close'] - df['close'].shift(1)) * (df['volume'] - df['volume'].shift(1))) < 0
    
    # Count the number of such divergent days in a window
    df['divergent_days_count'] = df['price_vol_divergence'].rolling(window=20).sum()
    
    # Determine if high volume days correspond with significant price movements
    df['price_vol_confirmation'] = ((df['volume'] > df['volume'].shift(1)) & (abs(df['close'] - df['close'].shift(1)) > df['ATR']))
    
    # Count the number of confirming days in a window
    df['confirming_days_count'] = df['price_vol_confirmation'].rolling(window=20).sum()
    
    # Calculate momentum using Volume-Averaged Price
    df['momentum'] = df['VAP'] - df['VAP'].shift(20)
    
    # Adjust momentum by averaged price range
    df['adjusted_momentum'] = df['momentum'] / df['avg_daily_price_range']
    
    # Generate alpha factors
    df['trend_factor'] = df['SMA'] - df['EMA']
    df['volatility_factor'] = df['ATR']
    df['volume_amount_factor'] = df['volume_to_amount_ratio']
    df['divergence_factor'] = df['divergent_days_count']
    df['adjusted_momentum_factor'] = df['adjusted_momentum']
    
    # Return the alpha factors
    return df[['trend_factor', 'volatility_factor', 'volume_amount_factor', 'divergence_factor', 'adjusted_momentum_factor']]
