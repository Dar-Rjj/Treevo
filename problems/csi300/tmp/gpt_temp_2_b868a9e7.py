import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Behavioral Divergence
    # Gap Efficiency Divergence
    short_term_gap = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    short_term_gap = short_term_gap.replace([np.inf, -np.inf], np.nan)
    
    medium_term_gap = np.abs(data['close'] - data['open'].shift(5))
    range_sum = (data['high'] - data['low']).rolling(window=6).sum()
    medium_term_gap = medium_term_gap / range_sum
    medium_term_gap = medium_term_gap.replace([np.inf, -np.inf], np.nan)
    
    gap_divergence = short_term_gap - medium_term_gap
    
    # Volatility Asymmetry
    upper_asymmetry = (data['high'] - data['open']) / (data['high'] - data['low'])
    lower_asymmetry = (data['open'] - data['low']) / (data['high'] - data['low'])
    upper_asymmetry = upper_asymmetry.replace([np.inf, -np.inf], np.nan)
    lower_asymmetry = lower_asymmetry.replace([np.inf, -np.inf], np.nan)
    
    volatility_asymmetry = upper_asymmetry - lower_asymmetry
    
    volatility_behavioral = gap_divergence * volatility_asymmetry
    
    # Entropy-Momentum Dynamics
    # Momentum Reversal
    short_term_momentum = (data['close'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    medium_term_momentum = (data['close'] - data['close'].shift(5)) / (data['high'].shift(5) - data['low'].shift(5))
    short_term_momentum = short_term_momentum.replace([np.inf, -np.inf], np.nan)
    medium_term_momentum = medium_term_momentum.replace([np.inf, -np.inf], np.nan)
    
    momentum_ratio = short_term_momentum / medium_term_momentum
    momentum_ratio = momentum_ratio.replace([np.inf, -np.inf], np.nan)
    
    # Behavioral Entropy
    price_entropy = np.sign(data['close'] - data['close'].shift(1)) * np.log(np.abs(data['close'] - data['close'].shift(1)) + 1)
    
    gap_ratio = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    gap_ratio = gap_ratio.replace([np.inf, -np.inf], np.nan)
    gap_entropy = -np.abs(data['close'] - data['open']) * np.log(gap_ratio + 1e-8)
    
    volume_entropy = np.sign(data['close'] - data['open']) * (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    volume_entropy = volume_entropy.replace([np.inf, -np.inf], np.nan)
    
    entropy_momentum = momentum_ratio * price_entropy * gap_entropy * volume_entropy
    
    # Volume-Pressure System
    # Volume Asymmetry
    returns = data['close'].pct_change()
    up_days = returns > 0
    down_days = returns < 0
    
    upside_volume_ratio = data['volume'].rolling(window=10).apply(
        lambda x: x[up_days.loc[x.index]].mean() if up_days.loc[x.index].any() else 0
    ) / data['volume'].rolling(window=10).mean()
    
    positive_returns_sum = returns.rolling(window=10).apply(
        lambda x: x[x > 0].sum() if (x > 0).any() else 0
    )
    negative_returns_sum = returns.rolling(window=10).apply(
        lambda x: x[x < 0].sum() if (x < 0).any() else 0
    )
    price_asymmetry = np.log(1 + positive_returns_sum) - np.log(1 + np.abs(negative_returns_sum))
    
    volume_asymmetry = upside_volume_ratio * price_asymmetry
    
    # Volume Dynamics
    turnover = data['volume'] * data['close']
    turnover_momentum = turnover / turnover.rolling(window=4).apply(lambda x: x.iloc[:-1].max() if len(x) > 1 else 1)
    turnover_momentum = turnover_momentum.replace([np.inf, -np.inf], np.nan)
    
    persistence = data['volume'] / data['volume'].shift(2)
    persistence = persistence.replace([np.inf, -np.inf], np.nan)
    
    stress = data['volume'] / (data['volume'].rolling(window=5).mean())
    stress = stress.replace([np.inf, -np.inf], np.nan)
    
    volume_pressure = volume_asymmetry * turnover_momentum * persistence / stress
    volume_pressure = volume_pressure.replace([np.inf, -np.inf], np.nan)
    
    # Breakout Detection
    # Volatility Break
    compression = (np.abs(data['close'] - data['open']).rolling(window=5).sum() / 
                  np.abs(data['close'] - data['open']).rolling(window=10).sum()) - 1
    compression = compression.replace([np.inf, -np.inf], np.nan)
    
    range_expansion = (data['high'].rolling(window=20).max() / 
                      data['low'].rolling(window=20).min()) - 1
    range_expansion = range_expansion.replace([np.inf, -np.inf], np.nan)
    
    true_range = np.maximum(data['high'] - data['low'], 
                           np.maximum(np.abs(data['high'] - data['close'].shift(1)), 
                                     np.abs(data['low'] - data['close'].shift(1))))
    avg_true_range = true_range.rolling(window=20).mean()
    
    # Behavioral Break
    efficiency = np.abs(data['close'] - data['open']) / (data['high'] - data['low'])
    efficiency = efficiency.replace([np.inf, -np.inf], np.nan)
    
    break_ratio = efficiency / efficiency.shift(1)
    break_ratio = break_ratio.replace([np.inf, -np.inf], np.nan)
    
    momentum_break = (data['close'] - data['high'].rolling(window=5).max()) / (
        data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    momentum_break = momentum_break.replace([np.inf, -np.inf], np.nan)
    
    breakout = momentum_break * (1 + np.abs(break_ratio)) * range_expansion / avg_true_range
    breakout = breakout.replace([np.inf, -np.inf], np.nan)
    
    # Market State Integration
    # Volatility State
    current_range = data['high'] - data['low']
    prev_range = data['high'].shift(5) - data['low'].shift(5)
    
    high_volatility = current_range > prev_range
    low_volatility = current_range < prev_range
    stable_volatility = ~(high_volatility | low_volatility)
    
    # Efficiency State
    behavioral_efficiency = efficiency
    high_efficiency = behavioral_efficiency > 0.6
    low_efficiency = behavioral_efficiency < 0.4
    medium_efficiency = ~(high_efficiency | low_efficiency)
    
    # State Selection
    base_factor = pd.Series(index=data.index, dtype=float)
    
    # High Volatility + High Efficiency
    condition1 = high_volatility & high_efficiency
    base_factor[condition1] = volatility_behavioral[condition1]
    
    # Low Volatility + Low Efficiency
    condition2 = low_volatility & low_efficiency
    base_factor[condition2] = volume_pressure[condition2]
    
    # Stable Volatility + Medium Efficiency
    condition3 = stable_volatility & medium_efficiency
    base_factor[condition3] = entropy_momentum[condition3]
    
    # Other cases - equal weight combination
    other_condition = ~(condition1 | condition2 | condition3)
    base_factor[other_condition] = (
        volatility_behavioral[other_condition] + 
        entropy_momentum[other_condition] + 
        volume_pressure[other_condition]
    ) / 3
    
    # Final Alpha
    enhanced_factor = base_factor * breakout
    volatility_compression = compression
    
    final_alpha = enhanced_factor * volatility_compression
    
    # Clean infinite values
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    
    return final_alpha
