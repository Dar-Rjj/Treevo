import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Measure Liquidity Conditions
    # Compute Bid-Ask Spread Proxy
    data['spread_proxy'] = (data['high'] - data['low']) / data['close']
    
    # Compute Volume Liquidity (Turnover Ratio)
    data['turnover_ratio'] = data['volume'] / (data['amount'] / data['close'])
    
    # Combine Liquidity Measures
    data['liquidity_measure'] = np.log(data['spread_proxy'] * data['turnover_ratio'] + 1e-8)
    
    # 2. Calculate Asymmetric Price Reversal
    # Compute 2-day returns for up/down move classification
    data['ret_2d'] = data['close'].pct_change(2)
    
    # Compute Up-Move Reversal (3-day return following 2-day up moves)
    data['up_move'] = (data['ret_2d'] > 0).astype(int)
    data['ret_3d_future'] = data['close'].pct_change(3).shift(-3)
    data['up_reversal'] = data['ret_3d_future'] * data['up_move'].shift(2)
    
    # Compute Down-Move Reversal (3-day return following 2-day down moves)
    data['down_move'] = (data['ret_2d'] < 0).astype(int)
    data['down_reversal'] = data['ret_3d_future'] * data['down_move'].shift(2)
    
    # Calculate Reversal Asymmetry
    data['reversal_asymmetry'] = (data['up_reversal'] - data['down_reversal']) * np.sign(data['close'].pct_change(1))
    
    # 3. Assess Volume Confirmation
    # Compute Volume Surprise
    data['volume_surprise'] = data['volume'] / data['volume'].rolling(window=10, min_periods=5).median()
    
    # Compute Price-Volume Divergence (5-day correlation)
    data['ret_5d'] = data['close'].pct_change(5)
    data['volume_change_5d'] = data['volume'].pct_change(5)
    data['price_volume_divergence'] = data['ret_5d'].rolling(window=5, min_periods=3).corr(data['volume_change_5d'])
    
    # Combine Volume Signals
    data['volume_confirmation'] = np.cbrt(data['volume_surprise'] * data['price_volume_divergence'])
    
    # 4. Liquidity-Weighted Signal Construction
    # Classify liquidity quintiles
    data['liquidity_rank'] = data['liquidity_measure'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.qcut(x, 5, labels=False, duplicates='drop')[-1] if len(x) >= 10 else 2, raw=False
    )
    
    # High Liquidity Processing (quintiles 4-5)
    high_liquidity_signal = data['reversal_asymmetry'] * data['volume_confirmation']
    
    # Low Liquidity Processing (quintiles 0-1)
    low_liquidity_signal = data['reversal_asymmetry'] * np.sqrt(np.abs(data['volume_confirmation'])) * np.sign(data['volume_confirmation'])
    
    # Signal Integration with liquidity regime adjustment
    data['liquidity_weighted_signal'] = np.where(
        data['liquidity_rank'] >= 3,  # High liquidity
        high_liquidity_signal,
        np.where(
            data['liquidity_rank'] <= 1,  # Low liquidity
            low_liquidity_signal,
            (high_liquidity_signal + low_liquidity_signal) / 2  # Medium liquidity
        )
    )
    
    # 5. Final Alpha Generation
    # Apply 2-day exponential smoothing
    alpha = data['liquidity_weighted_signal'].ewm(span=2, adjust=False).mean()
    
    # Apply signed power transformation
    alpha = np.sign(alpha) * np.power(np.abs(alpha), 1/3)
    
    return alpha
