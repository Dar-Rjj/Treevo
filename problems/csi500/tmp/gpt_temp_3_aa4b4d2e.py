import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Order Flow Imbalance Persistence Factor
    Calculates directional order flow imbalance and its persistence patterns to predict future returns
    """
    # Calculate daily returns for reference
    returns = df['close'].pct_change()
    
    # Calculate Directional Imbalance
    # Amount-based classification
    up_day_amount = np.where(returns > 0, df['amount'], 0)
    down_day_amount = np.where(returns < 0, df['amount'], 0)
    
    # Rolling averages for comparison
    up_amount_5d = pd.Series(up_day_amount, index=df.index).rolling(window=5).mean()
    down_amount_5d = pd.Series(down_day_amount, index=df.index).rolling(window=5).mean()
    
    # Amount per price move (efficiency measure)
    price_move = np.abs(returns)
    amount_per_move = np.where(price_move > 0, df['amount'] / price_move, 0)
    efficiency_ratio = amount_per_move / pd.Series(amount_per_move, index=df.index).rolling(window=10).mean()
    
    # Compute Net Imbalance
    net_imbalance = (up_amount_5d - down_amount_5d) / (up_amount_5d + down_amount_5d + 1e-8)
    directional_bias = net_imbalance * efficiency_ratio
    
    # Measure Persistence Patterns
    # Streak Analysis
    imbalance_direction = np.sign(directional_bias)
    streak_count = imbalance_direction * 0
    current_streak = 0
    current_sign = 0
    
    for i in range(len(imbalance_direction)):
        if i == 0:
            current_streak = 1
            current_sign = imbalance_direction.iloc[i] if not np.isnan(imbalance_direction.iloc[i]) else 0
        else:
            if imbalance_direction.iloc[i] == current_sign and current_sign != 0:
                current_streak += 1
            else:
                current_streak = 1
                current_sign = imbalance_direction.iloc[i] if not np.isnan(imbalance_direction.iloc[i]) else 0
        
        streak_count.iloc[i] = current_streak * current_sign
    
    # Streak magnitude progression
    streak_magnitude = directional_bias.rolling(window=3).apply(
        lambda x: np.mean(np.diff(x)) if len(x) == 3 and not any(np.isnan(x)) else 0
    )
    
    # Decay Patterns - imbalance reduction rates
    imbalance_ma_3 = directional_bias.rolling(window=3).mean()
    imbalance_ma_5 = directional_bias.rolling(window=5).mean()
    decay_rate = (imbalance_ma_3 - imbalance_ma_5) / (np.abs(imbalance_ma_5) + 1e-8)
    
    # Volatility impact adjustment
    volatility_5d = returns.rolling(window=5).std()
    vol_adjustment = 1 / (volatility_5d + 1e-8)
    
    # Generate Predictive Signal
    # Imbalance Strength Scoring
    current_imbalance_magnitude = np.abs(directional_bias)
    persistence_duration = np.abs(streak_count)
    
    # Combined strength score
    imbalance_strength = (current_imbalance_magnitude * 
                         persistence_duration * 
                         (1 - np.abs(decay_rate)) * 
                         vol_adjustment)
    
    # Direction application
    factor = directional_bias * imbalance_strength
    
    # Smooth the final factor
    factor = factor.rolling(window=3, min_periods=1).mean()
    
    return factor
