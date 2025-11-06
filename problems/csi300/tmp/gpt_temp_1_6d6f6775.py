import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume-Range Efficiency Momentum factor
    Combines momentum, efficiency, divergence, and amount-based signals
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic components
    returns = df['close'].pct_change()
    high_low_range = (df['high'] - df['low']) / df['close'].shift(1)
    daily_efficiency = np.abs(returns) / (high_low_range.replace(0, np.nan))
    
    # Multi-Period Momentum Analysis
    price_momentum_3 = df['close'].pct_change(3)
    price_momentum_5 = df['close'].pct_change(5)
    price_momentum_10 = df['close'].pct_change(10)
    
    volume_momentum_3 = df['volume'].pct_change(3)
    volume_momentum_5 = df['volume'].pct_change(5)
    volume_momentum_10 = df['volume'].pct_change(10)
    
    # Range Efficiency Analysis
    eff_ratio_5 = daily_efficiency.rolling(window=5, min_periods=3).mean()
    eff_ratio_10 = daily_efficiency.rolling(window=10, min_periods=5).mean()
    
    # Price-Volume-Range Divergence
    # Direction divergence (sign agreement)
    price_volume_dir_div = np.sign(returns) * np.sign(volume_momentum_3)
    
    # Magnitude divergence (normalized difference)
    norm_price_moves = returns.rolling(window=10, min_periods=5).std()
    norm_volume_moves = volume_momentum_3.rolling(window=10, min_periods=5).std()
    price_volume_mag_div = (np.abs(returns) / norm_price_moves.replace(0, np.nan)) - \
                          (np.abs(volume_momentum_3) / norm_volume_moves.replace(0, np.nan))
    
    # Range efficiency divergence
    eff_divergence = daily_efficiency - eff_ratio_5
    
    # Intraday Momentum Integration
    intraday_strength = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    intraday_efficiency = np.abs(intraday_strength)
    
    # Amount-Return Correlation
    amount_returns_corr = df['amount'].rolling(window=10, min_periods=5).corr(returns)
    amount_momentum = df['amount'].pct_change(5)
    
    # Alpha Factor Synthesis
    for i in range(len(df)):
        if i < 10:  # Ensure sufficient history
            alpha.iloc[i] = 0
            continue
            
        # Multi-dimensional divergence scoring
        divergence_score = (
            price_volume_dir_div.iloc[i] * 0.2 +
            price_volume_mag_div.iloc[i] * 0.3 +
            eff_divergence.iloc[i] * 0.2
        )
        
        # Momentum-efficiency integration
        momentum_score = (
            price_momentum_3.iloc[i] * 0.15 +
            price_momentum_5.iloc[i] * 0.1 +
            price_momentum_10.iloc[i] * 0.05 +
            (eff_ratio_5.iloc[i] - eff_ratio_10.iloc[i]) * 0.2
        )
        
        # Intraday component
        intraday_score = (
            intraday_strength.iloc[i] * 0.15 +
            (intraday_efficiency.iloc[i] - eff_ratio_5.iloc[i]) * 0.1
        )
        
        # Amount filtering and integration
        amount_filter = np.sign(amount_returns_corr.iloc[i]) if not pd.isna(amount_returns_corr.iloc[i]) else 1
        amount_component = amount_momentum.iloc[i] * 0.1
        
        # Final alpha calculation
        raw_alpha = (
            divergence_score * 0.4 +
            momentum_score * 0.3 +
            intraday_score * 0.2 +
            amount_component * 0.1
        ) * amount_filter
        
        alpha.iloc[i] = raw_alpha
    
    # Normalize the final alpha
    alpha = (alpha - alpha.rolling(window=20, min_periods=10).mean()) / \
            alpha.rolling(window=20, min_periods=10).std().replace(0, np.nan)
    
    return alpha.fillna(0)
