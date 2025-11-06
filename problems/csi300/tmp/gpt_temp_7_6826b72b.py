import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum with Volatility Scaling and Volume Regime Confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_2d'] = (data['close'] / data['close'].shift(2)) - 1
    data['momentum_5d'] = (data['close'] / data['close'].shift(5)) - 1
    data['momentum_15d'] = (data['close'] / data['close'].shift(15)) - 1
    data['momentum_40d'] = (data['close'] / data['close'].shift(40)) - 1
    
    # Volatility Scaling Framework
    # Range volatility (5-day)
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['vol_5d_range'] = data['daily_range'].rolling(window=5).mean()
    
    # Range volatility (20-day)
    data['vol_20d_range'] = data['daily_range'].rolling(window=20).mean()
    
    # Return volatility (10-day)
    data['daily_return'] = data['close'].pct_change()
    data['vol_10d_return'] = data['daily_return'].rolling(window=10).std()
    
    # True range volatility (5-day)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['vol_5d_true_range'] = data['true_range'].rolling(window=5).mean()
    
    # Volatility-scaled momentum
    data['scaled_momentum_2d'] = data['momentum_2d'] / (data['vol_5d_range'] + data['vol_10d_return'] + 1e-8)
    data['scaled_momentum_5d'] = data['momentum_5d'] / (data['vol_5d_range'] + data['vol_20d_range'] + 1e-8)
    data['scaled_momentum_15d'] = data['momentum_15d'] / (data['vol_20d_range'] + data['vol_10d_return'] + 1e-8)
    data['scaled_momentum_40d'] = data['momentum_40d'] / (data['vol_20d_range'] + data['vol_5d_true_range'] + 1e-8)
    
    # Volume Regime Analysis
    # Volume trend metrics
    data['volume_5d_ma'] = data['volume'].rolling(window=5).mean()
    data['volume_10d_ma'] = data['volume'].rolling(window=10).mean()
    data['volume_20d_ma'] = data['volume'].rolling(window=20).mean()
    
    data['volume_strength_ratio'] = data['volume_5d_ma'] / (data['volume_20d_ma'] + 1e-8)
    data['volume_acceleration'] = data['volume_5d_ma'] / (data['volume_10d_ma'] + 1e-8)
    
    # Volume stability coefficient
    data['volume_5d_std'] = data['volume'].rolling(window=5).std()
    data['volume_stability'] = data['volume_5d_std'] / (data['volume_5d_ma'] + 1e-8)
    
    # Volume persistence (correlation of volume ranks over 5 days)
    def volume_persistence(volume_series):
        if len(volume_series) < 5:
            return np.nan
        ranks = volume_series.rank()
        return ranks.corr(pd.Series(range(len(ranks)), index=ranks.index))
    
    data['volume_persistence'] = data['volume'].rolling(window=5).apply(volume_persistence, raw=False)
    
    # Volume regime classification
    data['high_momentum_regime'] = data['volume_acceleration'] > 1.15
    data['stable_regime'] = data['volume_stability'] < 0.25
    data['persistent_regime'] = data['volume_persistence'] > 0.6
    
    # Volume confirmation score
    data['volume_confirmation_score'] = data['volume_strength_ratio']
    data.loc[data['high_momentum_regime'], 'volume_confirmation_score'] *= 1.3
    data.loc[data['stable_regime'], 'volume_confirmation_score'] *= 1.2
    data.loc[data['persistent_regime'], 'volume_confirmation_score'] *= 1.1
    
    # Alpha Factor Construction
    # Weighted momentum score
    data['weighted_momentum'] = (
        0.3 * data['scaled_momentum_2d'] + 
        0.35 * data['scaled_momentum_5d'] + 
        0.25 * data['scaled_momentum_15d'] + 
        0.1 * data['scaled_momentum_40d']
    )
    
    # Momentum alignment bonus
    def momentum_alignment(row):
        momentums = [row['scaled_momentum_2d'], row['scaled_momentum_5d'], 
                    row['scaled_momentum_15d'], row['scaled_momentum_40d']]
        if all(m > 0 for m in momentums if not np.isnan(m)):
            return 1.15
        elif all(m < 0 for m in momentums if not np.isnan(m)):
            return 1.10
        else:
            return 0.85
    
    data['momentum_alignment_multiplier'] = data.apply(momentum_alignment, axis=1)
    data['aligned_momentum'] = data['weighted_momentum'] * data['momentum_alignment_multiplier']
    
    # Volume regime application
    data['momentum_with_volume'] = data['aligned_momentum'] * data['volume_confirmation_score']
    
    # Liquidity adjustment
    data['amount_5d_mean'] = data['amount'].rolling(window=5).mean()
    data['liquidity_adjustment'] = np.log(data['amount'] + 1e-8) / np.log(data['amount_5d_mean'] + 1e-8)
    
    # Final alpha factor
    data['alpha_factor'] = data['momentum_with_volume'] * data['liquidity_adjustment']
    
    return data['alpha_factor']
