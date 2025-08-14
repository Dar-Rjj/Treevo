def heuristics_v2(df):
    # Define the lookback periods
    short_lookback = 20
    long_lookback = 200
    pct_change_lookback = 5
    
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_20'] = df['close'].rolling(window=short_lookback).mean()
    
    # Compute Volume-Adjusted Volatility
    df['high_low_diff'] = df['high'] - df['low']
    df['volume_adjusted_volatility'] = df['high_low_diff'] * df['volume']
    df['volatility_rolling_avg'] = df['volume_adjusted_volatility'].rolling(window=short_lookback).mean()
    
    # Compute Price Momentum
    df['price_momentum'] = (df['close'] - df['SMA_20']) / df['close'].rolling(window=short_lookback).mean()
    
    # Incorporate Additional Price Change Metrics
    df['pct_change_5'] = df['close'].pct_change(periods=pct_change_lookback)
    df['high_low_range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['SMA_200'] = df['close'].rolling(window=long_lookback).mean()
    df['trend_indicator'] = df['SMA_20'] > df['SMA_200']
    df['trend_indicator'] = df['trend_indicator'].astype(int)
    
    # Incorporate Liquidity Measures
    df['daily_turnover'] = df['volume'] * df['close']
    df['turnover_rolling_avg'] = df['daily_turnover'].rolling(window=short_lookback).mean()
    
    # Define the weights for each component
    base_weights = {
        'price_momentum': 0.3,
        'volatility_rolling_avg': 0.2,
        'pct_change_5': 0.1,
        'trend_indicator': 0.2,
        'turnover_rolling_avg': 0.2
    }
    
    # Adjust weights based on market trend and liquidity
    def adjust_weights(row):
        if row['trend_indicator']:
            adjusted_weights = {k: v * 1.1 if k in ['price_momentum', 'pct_change_5'] else v for k, v in base_weights.items()}
            adjusted_weights = {k: v * 0.9 if k in ['volatility_rolling_avg', 'turnover_rolling_avg'] else v for k, v in adjusted_weights.items()}
        else:
            adjusted_weights = {k: v * 0.9 if k in ['price_momentum', 'pct_change_5'] else v for k, v in base_weights.items()}
            adjusted_weights = {k: v * 1.1 if k in ['volatility_rolling_avg', 'turnover_rolling_avg'] else v for k, v in adjusted_weights.items()}
        
        if row['turnover_rolling_avg'] > df['turnover_rolling_avg'].mean():
            adjusted_weights = {k: v * 1.1 if k in ['price_momentum', 'pct_change_5'] else v for k, v in adjusted_weights.items()}
            adjusted_weights = {k: v * 0.9 if k in ['volatility_rolling_avg', 'turnover_rolling_avg'] else v for k, v in adjusted_weights.items()}
        else:
            adjusted_weights = {k: v * 0.9 if k in ['price_momentum', 'pct_change_5'] else v for k, v in adjusted_weights.items()}
            adjusted_weights = {k: v * 1.1 if k in ['volatility_rolling_avg', 'turnover_rolling_avg'] else v for k, v in adjusted_weights.items()}
        
        return adjusted_weights
