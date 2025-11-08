import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Price-Volume Divergence Alpha Factor
    """
    df = data.copy()
    
    # Initialize EMA columns
    for period in [5, 10, 20]:
        df[f'ema_price_{period}d'] = np.nan
        df[f'ema_volume_{period}d'] = np.nan
    df['ema_range'] = np.nan
    
    # Calculate raw momentum
    for period in [5, 10, 20]:
        df[f'price_momentum_{period}d'] = (df['close'] / df['close'].shift(period)) - 1
        df[f'volume_momentum_{period}d'] = (df['volume'] / df['volume'].shift(period)) - 1
    
    # Calculate daily range
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    
    # Calculate amount trend
    df['amount_trend'] = (df['amount'] / df['amount'].shift(5)) - 1
    
    # Initialize regime columns
    df['volatility_regime'] = ''
    df['participation_regime'] = ''
    
    # Calculate EMAs and regimes iteratively
    alpha = 0.3
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for all calculations
            continue
            
        # Calculate EMAs
        for period in [5, 10, 20]:
            if pd.isna(df.loc[df.index[i-1], f'ema_price_{period}d']):
                df.loc[df.index[i], f'ema_price_{period}d'] = df.loc[df.index[i], f'price_momentum_{period}d']
                df.loc[df.index[i], f'ema_volume_{period}d'] = df.loc[df.index[i], f'volume_momentum_{period}d']
            else:
                df.loc[df.index[i], f'ema_price_{period}d'] = (
                    alpha * df.loc[df.index[i], f'price_momentum_{period}d'] + 
                    (1 - alpha) * df.loc[df.index[i-1], f'ema_price_{period}d']
                )
                df.loc[df.index[i], f'ema_volume_{period}d'] = (
                    alpha * df.loc[df.index[i], f'volume_momentum_{period}d'] + 
                    (1 - alpha) * df.loc[df.index[i-1], f'ema_volume_{period}d']
                )
        
        # Calculate range EMA
        if pd.isna(df.loc[df.index[i-1], 'ema_range']):
            df.loc[df.index[i], 'ema_range'] = df.loc[df.index[i], 'daily_range']
        else:
            df.loc[df.index[i], 'ema_range'] = (
                alpha * df.loc[df.index[i], 'daily_range'] + 
                (1 - alpha) * df.loc[df.index[i-1], 'ema_range']
            )
        
        # Determine regimes
        df.loc[df.index[i], 'volatility_regime'] = (
            'high' if df.loc[df.index[i], 'daily_range'] > df.loc[df.index[i], 'ema_range'] else 'low'
        )
        df.loc[df.index[i], 'participation_regime'] = (
            'high' if df.loc[df.index[i], 'amount_trend'] > 0 else 'low'
        )
    
    # Calculate cross-sectional ranks for each date
    df['price_rank'] = df.groupby(df.index)['ema_price_20d'].rank(pct=True)
    df['volume_rank'] = df.groupby(df.index)['ema_volume_20d'].rank(pct=True)
    
    # Calculate divergence score
    df['divergence'] = df['price_rank'] - df['volume_rank']
    
    # Calculate cross-sectional z-score of divergence
    divergence_mean = df.groupby(df.index)['divergence'].transform('mean')
    divergence_std = df.groupby(df.index)['divergence'].transform('std')
    df['divergence_z'] = (df['divergence'] - divergence_mean) / divergence_std
    
    # Initialize final alpha column
    df['alpha'] = np.nan
    
    # Calculate regime-adaptive weights and final alpha
    for i in range(len(df)):
        if i < 20:
            continue
            
        # Base regime weights
        if df.loc[df.index[i], 'volatility_regime'] == 'high':
            price_weight = 0.3
            volume_weight = 0.7
        else:  # low volatility
            price_weight = 0.7
            volume_weight = 0.3
        
        # Participation adjustment
        if df.loc[df.index[i], 'participation_regime'] == 'high':
            price_weight += 0.1
            volume_weight -= 0.1
        else:  # low participation
            price_weight -= 0.1
            volume_weight += 0.1
        
        # Ensure weights are within [0,1] bounds
        price_weight = max(0, min(1, price_weight))
        volume_weight = max(0, min(1, volume_weight))
        
        # Calculate weighted divergence for each timeframe
        weighted_divergences = []
        for period, timeframe_weight in [(5, 0.4), (10, 0.35), (20, 0.25)]:
            # Calculate ranks for this timeframe
            price_rank_period = df.loc[df.index[i], f'ema_price_{period}d']
            volume_rank_period = df.loc[df.index[i], f'ema_volume_{period}d']
            
            # Cross-sectional rank for this timeframe
            date_data = df[df.index == df.index[i]]
            price_rank_pct = (date_data[f'ema_price_{period}d'].rank(pct=True).iloc[0] 
                            if len(date_data) > 1 else 0.5)
            volume_rank_pct = (date_data[f'ema_volume_{period}d'].rank(pct=True).iloc[0] 
                             if len(date_data) > 1 else 0.5)
            
            # Weighted divergence for this timeframe
            weighted_price = price_weight * price_rank_pct
            weighted_volume = volume_weight * volume_rank_pct
            timeframe_divergence = weighted_price - weighted_volume
            
            weighted_divergences.append(timeframe_weight * timeframe_divergence)
        
        # Final alpha as weighted sum across timeframes
        df.loc[df.index[i], 'alpha'] = sum(weighted_divergences)
    
    return df['alpha']
