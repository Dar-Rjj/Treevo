import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, sp500_close):
    # Calculate Daily Log Return
    df['log_return'] = np.log(df['close']).diff()
    
    # Detect Market Regime
    # For simplicity, we use a moving average crossover to detect the market regime
    df['short_ma'] = df['close'].rolling(window=50).mean()
    df['long_ma'] = df['close'].rolling(window=200).mean()
    df['regime'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
    
    # Choose dynamic lookback period based on market regime
    df['lookback_period'] = np.where(df['regime'] == 1, 20, 60)
    
    # Price Momentum Score
    def ema_with_dynamic_lookback(series, lookback_series):
        ema = []
        for i in range(len(series)):
            if i < lookback_series.iloc[i]:
                ema.append(np.nan)
            else:
                ema.append(np.mean(series.iloc[i-lookback_series.iloc[i]:i]))
        return pd.Series(ema, index=series.index)
    
    df['price_momentum'] = ema_with_dynamic_lookback(df['log_return'], df['lookback_period'])
    
    # Volume Momentum Score
    df['volume_change'] = df['volume'].pct_change().fillna(0)
    df['volume_momentum'] = ema_with_dynamic_lookback(df['volume_change'], df['lookback_period'])
    
    # Combine Price and Volume Momentum Scores
    df['combined_momentum'] = df['price_momentum'] * df['volume_momentum']
    
    # Calculate Downside Volatility
    df['negative_returns'] = df['log_return'].apply(lambda x: x if x < 0 else 0)
    df['downside_volatility'] = np.sqrt((df['negative_returns'] ** 2).rolling(window=60).sum())
    
    # Adjust for Negative Impact
    df['adjusted_momentum'] = df['combined_momentum'] - df['downside_volatility']
    
    # Incorporate Relative Strength
    df['relative_strength'] = (df['close'] / sp500_close) - 1
    df['final_alpha'] = df['adjusted_momentum'] + df['relative_strength']
    
    # Final Alpha Factor
    final_alpha = df['final_alpha'].dropna()
    
    return final_alpha
