import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def heuristics_v2(df):
    # Calculate Daily VWAP
    df['daily_dollar_volume'] = df['volume'] * df['close']
    total_volume = df['volume'].sum()
    total_dollar_value = df['daily_dollar_volume'].sum()
    daily_vwap = total_dollar_value / total_volume
    
    # Calculate VWAP Deviation
    df['vwap_deviation'] = df['close'] - daily_vwap
    
    # Calculate Cumulative VWAP Deviation
    df['cumulative_vwap_deviation'] = df['vwap_deviation'].cumsum()
    
    # Integrate Short-Term Momentum (5 days)
    short_term_momentum = df['vwap_deviation'].rolling(window=5).sum()
    df['cumulative_vwap_deviation'] += short_term_momentum
    
    # Integrate Medium-Term Momentum (10 days)
    medium_term_momentum = df['vwap_deviation'].rolling(window=10).sum()
    df['cumulative_vwap_deviation'] += medium_term_momentum
    
    # Integrate Long-Term Momentum (20 days)
    long_term_momentum = df['vwap_deviation'].rolling(window=20).sum()
    df['cumulative_vwap_deviation'] += long_term_momentum
    
    # Calculate Intraday Volatility
    df['high_low_range'] = df['high'] - df['low']
    df['absolute_vwap_deviation'] = df['vwap_deviation'].abs()
    df['intraday_volatility'] = df['high_low_range'] + df['absolute_vwap_deviation']
    
    # Incorporate High-Frequency Data (Assume high-frequency data is available in the DataFrame)
    # Calculate High-Frequency VWAP
    df['hf_daily_dollar_volume'] = df['hf_volume'] * df['hf_price']
    hf_total_volume = df['hf_volume'].sum()
    hf_total_dollar_value = df['hf_daily_dollar_volume'].sum()
    hf_vwap = hf_total_dollar_value / hf_total_volume
    
    # Calculate High-Frequency VWAP Deviation
    df['hf_vwap_deviation'] = df['hf_close'] - hf_vwap
    
    # Use Machine Learning for Adaptive Weights
    features = df[['cumulative_vwap_deviation', 'short_term_momentum', 'medium_term_momentum', 'long_term_momentum', 'intraday_volatility']].dropna()
    target = df['future_returns'].shift(-1).dropna()
    model = LinearRegression()
    model.fit(features, target)
    adaptive_weights = model.predict(features)
    
    # Add Market Sentiment Analysis
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = [sentiment_analyzer.polarity_scores(text)['compound'] for text in df['news']]
    df['sentiment_score'] = sentiment_scores
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['cumulative_vwap_deviation'] * adaptive_weights[:, 0] +
                          short_term_momentum * adaptive_weights[:, 1] +
                          medium_term_momentum * adaptive_weights[:, 2] +
                          long_term_momentum * adaptive_weights[:, 3] +
                          df['intraday_volatility'] * adaptive_weights[:, 4] +
                          df['hf_vwap_deviation'] * 0.5 +  # Assume a fixed weight for high-frequency VWAP deviation
                          df['sentiment_score'])
    
    return df['alpha_factor']
