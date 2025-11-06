import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Extract columns
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    amount = df['amount']
    volume = df['volume']
    
    # Fractal Price-Volume Structure
    # Multi-Fractal Price Patterns
    price_fractal_divergence = ((high - low) / close) - ((high.shift(5) - low.shift(5)) / close.shift(5)) - ((high.shift(13) - low.shift(13)) / close.shift(13))
    intraday_price_fractal = ((close - open_price) * (high - low)) / (amount + 1e-6)
    price_fractal_dimension = price_fractal_divergence / (np.abs(intraday_price_fractal) + 1e-6)
    
    # Asymmetric Volume Dynamics
    prev_close = close.shift(1)
    volume_accumulation_strength = np.where(close > prev_close, (close - low) * amount, 0)
    volume_distribution_strength = np.where(close < prev_close, (high - close) * amount, 0)
    net_volume_direction = volume_accumulation_strength - volume_distribution_strength
    volume_asymmetry_ratio = volume_accumulation_strength / (volume_distribution_strength + 1e-6)
    
    # Price-Efficiency Dynamics
    # Price-Efficiency Distribution
    price_efficiency_range = amount / ((high - low) * (close - open_price) + 1e-6)
    gap_efficiency = ((close - open_price) * amount) / (np.abs(open_price - prev_close) + 1e-6)
    price_efficiency_divergence = (volume_accumulation_strength * amount / (amount.shift(3) + 1e-6)) - (volume_distribution_strength * amount / (amount.shift(3) + 1e-6))
    
    # Price-Efficiency Momentum Core
    prev_close_2 = close.shift(2)
    accumulation_momentum = np.where(close > prev_close, (close - prev_close) * price_efficiency_range, 0)
    distribution_momentum = np.where(close < prev_close, ((close - prev_close) - (prev_close - prev_close_2)) * price_efficiency_range, 0)
    price_efficiency_momentum_ratio = accumulation_momentum / (np.abs(distribution_momentum) + 1e-6)
    
    # Multi-Scale Asymmetry Detection
    # Price-Efficiency Asymmetry
    opening_gap_efficiency = (close - open_price) / (np.abs(open_price - prev_close) + 1e-6)
    intraday_range_asymmetry = (high - close) / (close - low + 1e-6)
    
    prev_open = open_price.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    price_efficiency_divergence_2 = np.sign(amount / (amount.shift(1) + 1e-6) - 1) * np.sign(
        (close - open_price) / (high - low + 1e-6) - (prev_close - prev_open) / (prev_high - prev_low + 1e-6)
    )
    
    # Multi-Scale Price-Volume Asymmetry
    short_term_asymmetry = ((close - prev_close) / (high - low + 1e-6)) * (amount / (amount.shift(1) + 1e-6))
    
    # Medium-term: 5-day average range
    medium_term_range = pd.Series([(high.iloc[max(0,i-4):i+1].max() - low.iloc[max(0,i-4):i+1].min()) 
                                  for i in range(len(high))], index=high.index)
    medium_term_asymmetry = (close - close.shift(5)) / (medium_term_range + 1e-6)
    
    # Long-term: 21-day max-min range
    long_term_range = pd.Series([(high.iloc[max(0,i-20):i+1].max() - low.iloc[max(0,i-20):i+1].min()) 
                               for i in range(len(high))], index=high.index)
    long_term_asymmetry = (close - close.shift(21)) / (long_term_range + 1e-6)
    
    multi_scale_price_volume_asymmetry = short_term_asymmetry + medium_term_asymmetry + long_term_asymmetry
    
    # Regime-Aware Factor Construction
    # Core Price Factors
    price_volume_fractal = price_fractal_divergence * net_volume_direction
    price_efficiency_dynamics = price_efficiency_divergence * price_efficiency_momentum_ratio
    asymmetry_price = intraday_range_asymmetry * volume_asymmetry_ratio
    gap_efficiency_price = gap_efficiency * opening_gap_efficiency
    
    # Dynamic Regime Weighting
    efficiency_regime = (price_efficiency_range > gap_efficiency).astype(float)
    volume_regime = (amount > amount.shift(1)).astype(float)
    asymmetry_regime = (intraday_range_asymmetry > 1).astype(float)
    
    # Multi-Fractal Integration
    # Regime-Specific Components
    high_volatility_alpha = price_volume_fractal * price_efficiency_dynamics * (1 + 0.4 * efficiency_regime)
    volume_price_alpha = asymmetry_price * price_efficiency_dynamics * (1 + 0.3 * volume_regime)
    efficiency_price_alpha = gap_efficiency_price * asymmetry_price * (1 + 0.2 * asymmetry_regime)
    
    # Multi-Scale Integration
    cross_price_base = (high_volatility_alpha + volume_price_alpha) * multi_scale_price_volume_asymmetry
    volume_enhanced_price_signal = cross_price_base * net_volume_direction
    range_weighted_price_alpha = volume_enhanced_price_signal * (high - low) * amount
    
    # Regime-Aware Finalization
    # Adaptive Components
    volatility_adaptive = ((close / prev_close - 1) / (high - low + 1e-6)) * (amount / (volume + 1e-6))
    price_level_adaptive = ((close - open_price) / (close + 1e-6)) * (amount / (amount.shift(1) + 1e-6))
    asymmetry_adjustment = 1 + 0.1 * (intraday_range_asymmetry - 1)
    
    # Dynamic Regime Multiplier
    dynamic_regime_multiplier = (1 + 0.4 * efficiency_regime) * (1 + 0.3 * volume_regime) * (1 + 0.2 * asymmetry_regime)
    
    # Final Alpha
    final_alpha = range_weighted_price_alpha * dynamic_regime_multiplier * volatility_adaptive * price_level_adaptive
    
    return final_alpha
