def heuristics_v2(df):
    # Calculate the rolling 10-day Pearson correlation between close price and volume
    correlation_pv = df[['close', 'volume']].rolling(window=10).corr().unstack().iloc[::2]['close'].droplevel(1)
    
    # Calculate the Rate of Change (ROC) for the closing price over 5 days
    roc_close = df['close'].pct_change(periods=5).fillna(0)
    
    # Combine the correlation and ROC to form the heuristics matrix
    heuristics_matrix = correlation_pv * roc_close
    
    return heuristics_matrix
