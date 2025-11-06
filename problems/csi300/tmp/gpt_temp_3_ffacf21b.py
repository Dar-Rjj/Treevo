import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum with Dynamic Volatility Scaling and Volume Acceleration
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Signals
    df['momentum_2d'] = df['close'] / df['close'].shift(2) - 1
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_15d'] = df['close'] / df['close'].shift(15) - 1
    df['momentum_40d'] = df['close'] / df['close'].shift(40) - 1
    
    # Dynamic Volatility Scaling
    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    df['vol_5d_range'] = df['daily_range_pct'].rolling(window=5).mean()
    df['vol_20d_range'] = df['daily_range_pct'].rolling(window=20).mean()
    
    # Volatility regime indicator
    df['vol_regime'] = df['vol_5d_range'] / (df['vol_20d_range'] + 0.0001)
    
    # Volatility-adjusted momentum
    df['mom_ultra_adj'] = df['momentum_2d'] / (df['vol_5d_range'] + 0.0001)
    df['mom_short_adj'] = df['momentum_5d'] / (df['vol_5d_range'] + 0.0001)
    df['mom_medium_adj'] = df['momentum_15d'] / (df['vol_20d_range'] + 0.0001)
    df['mom_long_adj'] = df['momentum_40d'] / (df['vol_20d_range'] + 0.0001)
    
    # Volume Acceleration Analysis
    df['volume_momentum_3d'] = df['volume'] / (df['volume'].shift(3) + 0.0001)
    df['volume_momentum_10d'] = df['volume'] / (df['volume'].shift(10) + 0.0001)
    df['volume_acceleration'] = (df['volume_momentum_3d'] - df['volume_momentum_10d']) / (df['volume_momentum_10d'] + 0.0001)
    df['volume_regime'] = df['volume_momentum_3d'] / (df['volume_momentum_10d'] + 0.0001)
    
    # Price-Volume Relationship
    df['daily_return'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    
    # Calculate rolling correlation between returns and volume changes
    price_volume_corr = []
    for i in range(len(df)):
        if i >= 9:
            window_returns = df['daily_return'].iloc[i-9:i+1]
            window_volume = df['volume_change'].iloc[i-9:i+1]
            corr = window_returns.corr(window_volume)
            price_volume_corr.append(corr if not np.isnan(corr) else 0)
        else:
            price_volume_corr.append(0)
    
    df['price_volume_corr'] = price_volume_corr
    df['volume_confirmation_signal'] = np.sign(df['price_volume_corr']) * df['volume_acceleration']
    
    # Alpha Factor Construction
    # Momentum combination with volatility regime weighting
    def get_momentum_combination(row):
        vol_regime = row['vol_regime']
        
        if vol_regime > 1.2:  # High volatility
            return (0.4 * row['mom_ultra_adj'] + 
                    0.3 * row['mom_short_adj'] + 
                    0.2 * row['mom_medium_adj'] + 
                    0.1 * row['mom_long_adj'])
        elif vol_regime < 0.8:  # Low volatility
            return (0.2 * row['mom_ultra_adj'] + 
                    0.3 * row['mom_short_adj'] + 
                    0.3 * row['mom_medium_adj'] + 
                    0.2 * row['mom_long_adj'])
        else:  # Normal volatility
            return (0.3 * row['mom_ultra_adj'] + 
                    0.3 * row['mom_short_adj'] + 
                    0.2 * row['mom_medium_adj'] + 
                    0.2 * row['mom_long_adj'])
    
    df['combined_momentum'] = df.apply(get_momentum_combination, axis=1)
    
    # Volume enhancement
    df['momentum_volume_enhanced'] = df['combined_momentum'] * (1 + df['volume_confirmation_signal'])
    
    # Liquidity adjustment
    df['amount_5d_avg'] = df['amount'].rolling(window=5).mean()
    df['liquidity_ratio'] = np.log(df['amount'] + 1) / (np.log(df['amount_5d_avg'] + 1) + 0.0001)
    
    # Final alpha factor
    alpha_factor = df['momentum_volume_enhanced'] * df['liquidity_ratio']
    
    return alpha_factor
