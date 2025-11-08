import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Aware Momentum-Volume Divergence Alpha Factor
    """
    df = data.copy()
    
    # Price Momentum Components
    df['price_mom_5'] = df['close'] / df['close'].shift(5) - 1
    df['price_mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['price_mom_20'] = df['close'] / df['close'].shift(20) - 1
    
    # Volume Momentum Components
    df['volume_mom_5'] = df['volume'] / df['volume'].shift(5) - 1
    df['volume_mom_10'] = df['volume'] / df['volume'].shift(10) - 1
    df['volume_mom_20'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Combined Momentum Scores (equal weighted average)
    df['price_momentum'] = (df['price_mom_5'] + df['price_mom_10'] + df['price_mom_20']) / 3
    df['volume_momentum'] = (df['volume_mom_5'] + df['volume_mom_10'] + df['volume_mom_20']) / 3
    
    # Regime Detection using Amount Data
    df['amount_mom_5'] = df['amount'] / df['amount'].shift(5) - 1
    df['amount_volatility'] = df['amount_mom_5'].rolling(window=20).std()
    
    # Regime Classification
    amount_mom_threshold = df['amount_mom_5'].rolling(window=20).quantile(0.7)
    amount_vol_threshold = df['amount_volatility'].rolling(window=20).quantile(0.7)
    
    df['high_activity'] = (df['amount_mom_5'] > amount_mom_threshold) & (df['amount_volatility'] > amount_vol_threshold)
    df['low_activity'] = (df['amount_mom_5'] <= amount_mom_threshold) & (df['amount_volatility'] <= amount_vol_threshold)
    df['transition'] = ~(df['high_activity'] | df['low_activity'])
    
    # Initialize EMA columns
    df['ema_price'] = np.nan
    df['ema_volume'] = np.nan
    
    # Exponential Smoothing with Regime-Adaptive Parameters
    for i in range(len(df)):
        if i == 0:
            df.loc[df.index[i], 'ema_price'] = df.loc[df.index[i], 'price_momentum']
            df.loc[df.index[i], 'ema_volume'] = df.loc[df.index[i], 'volume_momentum']
        else:
            if df.loc[df.index[i], 'high_activity']:
                alpha = 0.5
            elif df.loc[df.index[i], 'low_activity']:
                alpha = 0.1
            else:  # transition regime
                alpha = 0.3
            
            prev_ema_price = df.loc[df.index[i-1], 'ema_price']
            prev_ema_volume = df.loc[df.index[i-1], 'ema_volume']
            
            df.loc[df.index[i], 'ema_price'] = (alpha * df.loc[df.index[i], 'price_momentum'] + 
                                               (1 - alpha) * prev_ema_price)
            df.loc[df.index[i], 'ema_volume'] = (alpha * df.loc[df.index[i], 'volume_momentum'] + 
                                                (1 - alpha) * prev_ema_volume)
    
    # Divergence Pattern Analysis
    df['strength_ratio'] = df['ema_price'] / (df['ema_volume'] + 1e-8)  # avoid division by zero
    
    # Acceleration Difference
    df['price_acceleration'] = df['ema_price'] - df['ema_price'].shift(1)
    df['volume_acceleration'] = df['ema_volume'] - df['ema_volume'].shift(1)
    df['acceleration_diff'] = df['price_acceleration'] - df['volume_acceleration']
    
    # Directional Divergence Score
    df['directional_divergence'] = np.where(
        (df['price_momentum'] > 0) & (df['volume_momentum'] < 0), 1,  # Bullish divergence
        np.where((df['price_momentum'] < 0) & (df['volume_momentum'] > 0), -1, 0)  # Bearish divergence
    )
    
    # Combined Divergence Score
    df['divergence_score'] = (df['strength_ratio'] * df['directional_divergence'] + 
                             df['acceleration_diff'] * 0.5)
    
    # Cross-Sectional Ranking within Regimes
    df['regime_group'] = np.where(df['high_activity'], 'high', 
                                 np.where(df['low_activity'], 'low', 'transition'))
    
    # Calculate percentile ranks within each regime group
    df['divergence_rank'] = df.groupby('regime_group')['divergence_score'].transform(
        lambda x: x.rank(pct=True) * 100
    )
    
    # Dynamic Weight Adjustment based on regime
    df['regime_weight'] = np.where(
        df['high_activity'], 0.7,  # Higher weight to volume confirmation
        np.where(df['low_activity'], 0.3,  # Higher weight to price momentum
                0.5)  # Equal weighting for transition
    )
    
    # Final Alpha Score
    df['alpha_score'] = df['regime_weight'] * df['divergence_rank']
    
    return df['alpha_score']
