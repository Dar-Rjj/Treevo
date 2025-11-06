import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Alpha 1: Momentum Divergence with Volume Confirmation
    # Calculate Price Momentum
    short_term_momentum = data['close'] / data['close'].shift(5) - 1
    medium_term_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Calculate Momentum Divergence
    momentum_divergence = np.abs(short_term_momentum - medium_term_momentum)
    
    # Volume Confirmation
    avg_volume_20 = data['volume'].shift(1).rolling(window=20).mean()
    volume_ratio = data['volume'] / avg_volume_20
    alpha1 = momentum_divergence * volume_ratio
    
    # Alpha 2: Volatility Regime Adjusted Return
    # Calculate Recent Return
    recent_return = data['close'] / data['close'].shift(5) - 1
    
    # Assess Volatility Regime
    current_range = (data['high'] - data['low']) / data['close']
    avg_range_20 = ((data['high'] - data['low']) / data['close']).shift(1).rolling(window=20).mean()
    volatility_ratio = current_range / avg_range_20
    
    # Adjust Return by Volatility
    alpha2 = recent_return / volatility_ratio
    alpha2 = np.sign(recent_return) * np.abs(alpha2)
    
    # Alpha 3: Liquidity-Adjusted Price Reversal
    # Identify Large Price Moves
    daily_return = data['close'] / data['close'].shift(1) - 1
    large_moves = daily_return.copy()
    large_moves[np.abs(daily_return) <= 0.02] = 0
    
    # Assess Liquidity Conditions
    volume_to_amount = data['volume'] / data['amount']
    avg_volume_amount_20 = (data['volume'] / data['amount']).shift(1).rolling(window=20).mean()
    liquidity_ratio = volume_to_amount / avg_volume_amount_20
    
    # Compute Reversal Signal
    alpha3 = -large_moves * liquidity_ratio
    
    # Alpha 4: Intraday Strength Persistence
    # Calculate Intraday Strength
    intraday_strength = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    intraday_strength = 2 * intraday_strength - 1  # Scale to -1 to 1
    
    # Assess Persistence
    def calculate_streak(strength_series):
        streak = pd.Series(0, index=strength_series.index)
        current_streak = 0
        current_sign = 0
        
        for i in range(len(strength_series)):
            if i == 0:
                streak.iloc[i] = 1
                current_sign = np.sign(strength_series.iloc[i]) if strength_series.iloc[i] != 0 else 0
                current_streak = 1
            else:
                current_sign_val = np.sign(strength_series.iloc[i]) if strength_series.iloc[i] != 0 else 0
                if current_sign_val == current_sign and current_sign_val != 0:
                    current_streak += 1
                else:
                    current_streak = 1
                    current_sign = current_sign_val
                
                # Apply decay for longer streaks
                decay_factor = 1.0 / (1 + 0.1 * (current_streak - 1))
                streak.iloc[i] = current_streak * decay_factor
        
        return streak
    
    # Use last 6 days for streak calculation
    strength_window = intraday_strength.rolling(window=6).apply(
        lambda x: calculate_streak(pd.Series(x))[-1] if len(x) == 6 else 0, raw=False
    )
    
    persistence_magnitude = intraday_strength * strength_window
    
    # Volume Validation
    avg_volume_5 = data['volume'].shift(1).rolling(window=5).mean()
    volume_trend = data['volume'] / avg_volume_5
    
    alpha4 = persistence_magnitude * volume_trend
    
    # Alpha 5: Amount-Based Momentum Acceleration
    # Calculate Price Momentum
    price_momentum_10 = data['close'] / data['close'].shift(10) - 1
    price_momentum_5 = data['close'] / data['close'].shift(5) - 1
    
    # Calculate Amount Momentum
    avg_amount_10 = data['amount'].shift(1).rolling(window=10).mean()
    amount_momentum_current = data['amount'] / avg_amount_10 - 1
    
    avg_amount_5_prev = data['amount'].shift(5).rolling(window=10).mean()
    amount_momentum_prev = data['amount'].shift(5) / avg_amount_5_prev - 1
    
    # Compute Acceleration Signal
    price_acceleration = price_momentum_10 - price_momentum_5
    amount_acceleration = amount_momentum_current - amount_momentum_prev
    
    # Combine Signals
    raw_alpha5 = price_acceleration * amount_acceleration
    alpha5 = raw_alpha5.rolling(window=3).mean()
    
    # Combine all alphas with equal weighting
    factors = pd.DataFrame({
        'alpha1': alpha1,
        'alpha2': alpha2,
        'alpha3': alpha3,
        'alpha4': alpha4,
        'alpha5': alpha5
    })
    
    # Handle NaN values and normalize
    factors = factors.fillna(0)
    final_factor = factors.mean(axis=1)
    
    return final_factor
