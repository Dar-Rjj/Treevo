import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Momentum Contribution
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['momentum_contribution'] = df['volume'] * df['price_change'].abs()

    # Integrate Historical Momentum Contributions
    df['sum_momentum_contributions'] = df['momentum_contribution'].rolling(window=5).sum()
    df['max_min_price_changes'] = df['price_change'].rolling(window=5).max() - df['price_change'].rolling(window=5).min()
    df['historical_momentum'] = df['sum_momentum_contributions'] * (df['max_min_price_changes'] > 0)

    # Adjust for Market Sentiment
    df['volatility_threshold'] = ((df['high'] - df['low']) / df['close']).rolling(window=5).mean()
    df['adjusted_avmi'] = df['historical_momentum'].apply(lambda x: x * 1.1 if x > df['volatility_threshold'] else x * 0.9)

    # Calculate Volume-Adjusted Price Momentum
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['sum_volume_changes'] = df['volume_change'].rolling(window=5).sum()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['price_momentum'] = df['log_return'] * (df['sum_volume_changes'] > 0)

    # Evaluate Intraday and Overnight Dynamics
    df['intraday_return_ratio_high_low'] = df['high'] / df['low']
    df['intraday_return_ratio_close_open'] = df['close'] / df['open']
    df['overnight_return'] = np.log(df['open'] / df['close'].shift(1))
    df['log_volume'] = np.log(df['volume'])

    # Introduce Intraday Intensity
    df['intraday_trading_range'] = df['high'] - df['low']
    df['intraday_volatility'] = df['intraday_trading_range'] * df['volume']
    df['intraday_trading_activity'] = df['intraday_volatility'] * df['volume']

    # Synthesize Hybrid Momentum, Reversal, and Intraday Intensity Signals
    df['weighted_intraday_return'] = (df['intraday_return_ratio_high_low'] + df['intraday_return_ratio_close_open']) / 2
    df['hybrid_signal'] = (df['weighted_intraday_return'] - df['overnight_return'] + df['intraday_trading_activity']) * df['adjusted_avmi']

    return df['hybrid_signal']
