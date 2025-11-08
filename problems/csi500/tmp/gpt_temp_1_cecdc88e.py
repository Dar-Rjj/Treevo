import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Momentum-Volume Divergence Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Raw Momentum and Volume Calculation
    # Price Momentum Components
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['price_momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['price_momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Volume Momentum Components
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_10d'] = data['volume'] / data['volume'].shift(10) - 1
    data['volume_momentum_20d'] = data['volume'] / data['volume'].shift(20) - 1
    
    # Exponential Smoothing Application
    # Smoothed Price Momentum
    data['ema_price_5d'] = data['price_momentum_5d'].ewm(alpha=0.3, adjust=False).mean()
    data['ema_price_10d'] = data['price_momentum_10d'].ewm(alpha=0.3, adjust=False).mean()
    data['ema_price_20d'] = data['price_momentum_20d'].ewm(alpha=0.3, adjust=False).mean()
    
    # Smoothed Volume Momentum
    data['ema_volume_5d'] = data['volume_momentum_5d'].ewm(alpha=0.3, adjust=False).mean()
    data['ema_volume_10d'] = data['volume_momentum_10d'].ewm(alpha=0.3, adjust=False).mean()
    data['ema_volume_20d'] = data['volume_momentum_20d'].ewm(alpha=0.3, adjust=False).mean()
    
    # Regime Detection Using Amount Data
    # Amount-Based Regime Indicator
    data['amount_trend_10d'] = data['amount'] / data['amount'].shift(10) - 1
    data['amount_volatility'] = data['amount'].pct_change().rolling(window=20).std()
    
    # Regime Classification
    amount_trend_threshold = data['amount_trend_10d'].rolling(window=50).quantile(0.7)
    amount_vol_threshold = data['amount_volatility'].rolling(window=50).quantile(0.7)
    
    data['regime'] = 0  # Default: Transition regime
    data.loc[(data['amount_trend_10d'] > amount_trend_threshold) & 
             (data['amount_volatility'] > amount_vol_threshold), 'regime'] = 1  # High Activity
    data.loc[(data['amount_trend_10d'] <= amount_trend_threshold.rolling(window=5).mean()) & 
             (data['amount_volatility'] <= amount_vol_threshold.rolling(window=5).mean()), 'regime'] = -1  # Low Activity
    
    # Momentum-Volume Divergence Calculation
    # Directional Divergence
    data['directional_div_5d'] = np.sign(data['ema_price_5d']) * np.sign(data['ema_volume_5d'])
    data['directional_div_10d'] = np.sign(data['ema_price_10d']) * np.sign(data['ema_volume_10d'])
    data['directional_div_20d'] = np.sign(data['ema_price_20d']) * np.sign(data['ema_volume_20d'])
    
    # Multi-timeframe Agreement Score
    data['agreement_score'] = (data['directional_div_5d'] + data['directional_div_10d'] + data['directional_div_20d']) / 3
    
    # Magnitude Divergence
    data['magnitude_div_5d'] = data['ema_price_5d'] - data['ema_volume_5d']
    data['magnitude_div_10d'] = data['ema_price_10d'] - data['ema_volume_10d']
    data['magnitude_div_20d'] = data['ema_price_20d'] - data['ema_volume_20d']
    
    # Acceleration Difference
    data['acceleration_diff_5d'] = data['ema_price_5d'].diff(5) - data['ema_volume_5d'].diff(5)
    data['acceleration_diff_10d'] = data['ema_price_10d'].diff(10) - data['ema_volume_10d'].diff(10)
    data['acceleration_diff_20d'] = data['ema_price_20d'].diff(20) - data['ema_volume_20d'].diff(20)
    
    # Cross-Sectional Ranking Application
    # Daily Cross-Sectional Percentile Ranking
    for col in ['ema_price_5d', 'ema_price_10d', 'ema_price_20d', 
                'ema_volume_5d', 'ema_volume_10d', 'ema_volume_20d',
                'agreement_score', 'magnitude_div_5d', 'magnitude_div_10d', 'magnitude_div_20d']:
        data[f'rank_{col}'] = data.groupby(data.index)[col].transform(lambda x: x.rank(pct=True))
    
    # Dynamic Weighting Based on Regime
    data['weight_price'] = 0.5  # Default for Transition regime
    data['weight_volume'] = 0.5  # Default for Transition regime
    
    # High Activity: Higher weight to Volume Confirmation
    data.loc[data['regime'] == 1, 'weight_price'] = 0.3
    data.loc[data['regime'] == 1, 'weight_volume'] = 0.7
    
    # Low Activity: Higher weight to Price Momentum
    data.loc[data['regime'] == -1, 'weight_price'] = 0.7
    data.loc[data['regime'] == -1, 'weight_volume'] = 0.3
    
    # Factor Output Generation
    # Combined Signal Calculation
    price_component = (data['rank_ema_price_5d'] + data['rank_ema_price_10d'] + data['rank_ema_price_20d']) / 3
    volume_component = (data['rank_ema_volume_5d'] + data['rank_ema_volume_10d'] + data['rank_ema_volume_20d']) / 3
    divergence_component = (data['rank_agreement_score'] + 
                           data['rank_magnitude_div_5d'] + 
                           data['rank_magnitude_div_10d'] + 
                           data['rank_magnitude_div_20d']) / 4
    
    # Regime-Weighted Divergence Score
    data['regime_weighted_signal'] = (data['weight_price'] * price_component + 
                                     data['weight_volume'] * volume_component + 
                                     divergence_component) / 3
    
    # Volatility Scaling
    price_volatility = data['close'].pct_change().rolling(window=20).std()
    data['final_factor'] = data['regime_weighted_signal'] / price_volatility
    
    # Return the final factor series
    return data['final_factor']
