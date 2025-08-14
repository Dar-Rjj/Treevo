import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, market_index_close, fixed_window=60, volatility_threshold=0.05, volume_threshold=1e6, momentum_period=20):
    # Calculate Close-to-Open Return
    df['NextDayOpen'] = df['open'].shift(-1)
    df['CloseToOpenReturn'] = (df['NextDayOpen'] - df['close']) / df['close']

    # Volume Weighting
    df['VolumeWeightedReturn'] = df['CloseToOpenReturn'] * df['volume']

    # Determine Volatility
    df['Volatility'] = df[['high', 'low', 'close']].rolling(window=fixed_window).std().mean(axis=1)

    # Adaptive Window Calculation
    df['AdaptiveWindow'] = np.where(df['Volatility'] > volatility_threshold, fixed_window // 2, fixed_window * 2)
    
    # Liquidity Consideration
    df['AverageVolume'] = df['volume'].rolling(window=fixed_window).mean()
    df['AdaptiveWindow'] = np.where(df['AverageVolume'] > volume_threshold, df['AdaptiveWindow'] // 2, df['AdaptiveWindow'] * 2)

    # Momentum Component
    df['Momentum'] = df['close'].pct_change(periods=momentum_period)
    df['MomentumAdjustedWeight'] = np.where(df['Momentum'] > 0, 1.5, 0.5)

    # Cross-Asset Correlation
    df['MarketCorrelation'] = df['close'].rolling(window=fixed_window).corr(market_index_close)
    df['CorrelationAdjustedWeight'] = np.where(df['MarketCorrelation'] > 0.5, 1.5, 0.5)

    # Final Factor Calculation
    df['FactorValue'] = (df['VolumeWeightedReturn']
                         .rolling(window=df['AdaptiveWindow'], min_periods=1)
                         .mean() * df['MomentumAdjustedWeight'] * df['CorrelationAdjustedWeight'])

    return df['FactorValue']

# Example usage:
# df = pd.read_csv('your_data.csv')
# market_index_close = pd.read_csv('market_index_data.csv')['close']
# factor_values = heuristics_v2(df, market_index_close)
