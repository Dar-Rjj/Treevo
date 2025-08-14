import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Range (High - Low) / Open
    df['intraday_range'] = (df['high'] - df['low']) / df['open']

    # Calculate Close-to-Previous-Day-Close Return
    df['prev_close'] = df['close'].shift(1)
    df['close_to_prev_return'] = (df['close'] - df['prev_close']) / df['prev_close']

    # Assign weights based on the age of the data
    recent_weights = {'intraday_range': 0.7, 'close_to_prev_return': 0.3}
    older_weights = {'intraday_range': 0.5, 'close_to_prev_return': 0.5}

    # Combine Intraday Range and Close-to-Previous-Day-Close Return with weights
    df['alpha_factor'] = 0
    for idx, row in df.iterrows():
        if pd.notna(row['prev_close']):  # Ensure we have a previous close value
            weight = recent_weights if len(df.loc[:idx]) < len(df) * 0.5 else older_weights
            df.at[idx, 'alpha_factor'] = (weight['intraday_range'] * row['intraday_range'] +
                                          weight['close_to_prev_return'] * row['close_to_prev_return'])

    return df['alpha_factor']
