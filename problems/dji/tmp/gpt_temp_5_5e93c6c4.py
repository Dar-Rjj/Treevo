import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate the intraday range (High - Low)
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate the ratio of Close to Open price
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Calculate the change in volume from the previous day
    df['volume_change'] = df['volume'].diff()
    
    # Calculate the ratio of volume to average volume over the last 5 days
    df['avg_volume_5d'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['avg_volume_5d']
    
    # Calculate the trend of the Close price over the past 5 days
    df['close_trend'] = df['close'].rolling(window=5).apply(lambda x: 1 if x[-1] > x[0] else -1, raw=True)
    
    # Compare the amount traded today with the amount traded yesterday
    df['amount_change'] = df['amount'].diff()
    
    # Calculate the ratio of amount to volume
    df['amount_to_volume_ratio'] = df['amount'] / df['volume']
    
    # Combine the indicators into a composite score
    def composite_score(row):
        score = 0
        # Intraday Range
        if row['intraday_range'] > row['intraday_range'].shift(1):
            score += 1
        else:
            score -= 1
        # Close/Open Ratio
        if row['close_open_ratio'] > 1:
            score += 1
        elif row['close_open_ratio'] < 1:
            score -= 1
        # Volume Change
        if row['volume_change'] > 0:
            score += 1
        else:
            score -= 1
        # Volume Ratio
        if row['volume_ratio'] > 1.5:
            score += 1
        else:
            score -= 1
        # Close Trend
        if row['close_trend'] == 1:
            score += 1
        elif row['close_trend'] == -1:
            score -= 1
        # Amount Change
        if row['amount_change'] > 0:
            score += 1
        else:
            score -= 1
        # Amount to Volume Ratio
        if row['amount_to_volume_ratio'] > 1:
            score += 1
        else:
            score -= 1
        return score
    
    df['composite_score'] = df.apply(composite_score, axis=1)
    
    # Use the composite score as the final alpha factor
    alpha_factor = df['composite_score']
    
    return alpha_factor
