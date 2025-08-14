import pandas as pd
    from textblob import TextBlob

    # Assuming 'news' is one of the columns in df containing textual data
    def get_sentiment(text):
        return TextBlob(text).sentiment.polarity
    
    df['momentum'] = df['close'].pct_change(12*20)  # Assuming 20 trading days per month
    df['sentiment'] = df['news'].apply(get_sentiment)
    
    # Simple equal weighting for demonstration
    df['heuristic_value'] = (df['momentum'] + df['sentiment']) / 2
    heuristics_matrix = df['heuristic_value']
    
    return heuristics_matrix
