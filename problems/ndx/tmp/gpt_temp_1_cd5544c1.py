defined financial ratios and price/volume transformations, aiming to capture different aspects of market dynamics without normalization.}

```python
def heuristics_v2(df):
    # Momentum factor: (Close - Close 30 days ago) / Close 30 days ago
    momentum = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
    
    # Reversal factor: (Close 5 days ago - Close) / Close 5 days ago
    reversal = (df['close'].shift(5) - df['close']) / df['close'].shift(5)
    
    # Volatility factor: Standard deviation of log returns over the last 30 days
    log_returns = (df['close']/df['close'].shift(1)).apply(np.log)
    volatility = log_returns.rolling(window=30).std()
    
    # Volume shock: (Volume - Volume 10 days moving average) / Volume 10 days moving average
    volume_ma = df['volume'].rolling(window=10).mean()
    volume_shock = (df['volume'] - volume_ma) / volume_ma
    
    # Price range expansion: (High - Low) / Open
    price_range = (df['high'] - df['low']) / df['open']
    
    heuristics_matrix = pd.DataFrame({'momentum': momentum, 'reversal': reversal, 
                                      'volatility': volatility, 'volume_shock': volume_shock, 
                                      'price_range': price_range})
    
    return heuristics_matrix
