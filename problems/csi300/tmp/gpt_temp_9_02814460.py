import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Helper function for fractal true range
    def fractal_true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(np.abs(high - close_prev), np.abs(low - close_prev)))
    
    # Multi-Timeframe Fractal Momentum
    # 5-day Fractal Momentum
    def calculate_fractal_momentum_5d(high, low, close):
        fractal_scores = []
        for i in range(len(close)):
            if i < 4:
                fractal_scores.append(0)
                continue
            window_high = high.iloc[i-4:i+1]
            window_low = low.iloc[i-4:i+1]
            window_close = close.iloc[i-4:i+1]
            
            # Calculate price path complexity
            high_low_range = (window_high - window_low) / window_close
            close_to_close = window_close.pct_change().abs()
            
            # Directional persistence vs noise
            directional_moves = (window_close.diff() > 0).astype(int)
            persistence_score = directional_moves.rolling(window=3).mean().iloc[-1] if len(directional_moves) >= 3 else 0.5
            
            # Fractal momentum score
            complexity_score = high_low_range.mean() / (close_to_close.mean() + 1e-8)
            momentum_score = (window_close.iloc[-1] / window_close.iloc[0] - 1) * complexity_score * persistence_score
            fractal_scores.append(momentum_score)
        
        return pd.Series(fractal_scores, index=close.index)
    
    # 20-day Fractal Momentum
    def calculate_fractal_momentum_20d(high, low, close):
        fractal_scores = []
        for i in range(len(close)):
            if i < 19:
                fractal_scores.append(0)
                continue
            window_high = high.iloc[i-19:i+1]
            window_low = low.iloc[i-19:i+1]
            window_close = close.iloc[i-19:i+1]
            
            # Analyze price pattern self-similarity across sub-windows
            sub_windows = [window_close.iloc[j:j+5] for j in range(0, 16, 5)]
            trend_consistency = 0
            if len(sub_windows) >= 3:
                trends = [1 if sub_windows[k].iloc[-1] > sub_windows[k].iloc[0] else -1 for k in range(len(sub_windows))]
                trend_consistency = sum(trends) / len(trends)
            
            # Fractal dimension momentum
            price_range = (window_high.max() - window_low.min()) / window_close.mean()
            volatility = window_close.pct_change().std()
            fractal_score = price_range / (volatility + 1e-8) * trend_consistency
            fractal_scores.append(fractal_score)
        
        return pd.Series(fractal_scores, index=close.index)
    
    # Calculate fractal momentums
    fractal_momentum_5d = calculate_fractal_momentum_5d(data['high'], data['low'], data['close'])
    fractal_momentum_20d = calculate_fractal_momentum_20d(data['high'], data['low'], data['close'])
    
    # Fractal Momentum Acceleration Signal
    fractal_momentum_acceleration = (fractal_momentum_5d - fractal_momentum_20d) / (fractal_momentum_5d.abs() + 1e-8)
    
    # Gap-Intraday Fractal Analysis
    gap_fractal_pct = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    gap_fractal_5d_momentum = gap_fractal_pct.rolling(window=5).mean()
    gap_fractal_acceleration = gap_fractal_5d_momentum.diff()
    
    intraday_fractal_return = (data['close'] - data['open']) / (data['open'] + 1e-8)
    intraday_fractal_5d_momentum = intraday_fractal_return.rolling(window=5).mean()
    intraday_fractal_acceleration = intraday_fractal_5d_momentum.diff()
    
    # Dynamic Liquidity Absorption Acceleration
    # Volume Fractal Analysis
    volume_fractal_5d_return = data['volume'] / data['volume'].shift(5) - 1
    volume_fractal_20d_return = data['volume'] / data['volume'].shift(20) - 1
    volume_fractal_trend_direction = np.sign(volume_fractal_5d_return - volume_fractal_20d_return)
    
    # Fractal Divergence Patterns
    bullish_fractal_divergence = (fractal_momentum_5d < 0) & (volume_fractal_5d_return > 0)
    bearish_fractal_divergence = (fractal_momentum_5d > 0) & (volume_fractal_5d_return < 0)
    volume_fractal_confirmation = np.sign(fractal_momentum_5d) == np.sign(volume_fractal_5d_return)
    
    # Amount Fractal Analysis
    amount_fractal_acceleration = data['amount'].pct_change()
    volume_amount_fractal_divergence = np.sign(volume_fractal_5d_return) != np.sign(amount_fractal_acceleration)
    liquidity_absorption_memory = (data['volume'] / data['amount']).rolling(window=5).std()
    
    # Range Expansion Asymmetry
    upside_range_expansion = (data['high'] - data['close']) / data['close']
    downside_range_compression = (data['close'] - data['low']) / data['close']
    
    upside_expansion_acceleration = upside_range_expansion.rolling(window=5).mean().diff()
    downside_compression_acceleration = downside_range_compression.rolling(window=5).mean().diff()
    
    range_asymmetry_acceleration = upside_expansion_acceleration - downside_compression_acceleration
    
    # Volume Confirmation Acceleration
    expansion_volume_support = (data['volume'] * (upside_range_expansion > 0)).rolling(window=5).mean()
    compression_volume_patterns = (data['volume'] * (downside_range_compression > 0)).rolling(window=5).mean()
    
    volume_confirmed_asymmetry = range_asymmetry_acceleration * (expansion_volume_support - compression_volume_patterns)
    
    # Multi-Scale Volatility Regime
    close_prev = data['close'].shift(1)
    fractal_true_range_vals = fractal_true_range(data['high'], data['low'], close_prev)
    short_term_fractal_vol = (fractal_true_range_vals / data['close']).rolling(window=5).mean()
    medium_term_fractal_vol = ((data['high'] - data['low']) / data['close']).rolling(window=20).mean()
    
    high_fractal_vol_regime = short_term_fractal_vol > (1.3 * medium_term_fractal_vol)
    low_fractal_vol_regime = short_term_fractal_vol < (0.8 * medium_term_fractal_vol)
    
    # Microstructure Memory Acceleration
    signed_volume_persistence = (data['volume'] * np.sign(data['close'].diff())).rolling(window=5).mean()
    order_flow_memory_acceleration = signed_volume_persistence.diff()
    
    price_impact_memory = (data['close'].diff().abs() / data['volume']).rolling(window=5).mean()
    price_impact_memory_acceleration = -price_impact_memory.diff()  # Negative because lower impact is better
    
    microstructure_memory_acceleration = order_flow_memory_acceleration + price_impact_memory_acceleration
    
    # Cross-Dimensional Fractal Divergence Integration
    gap_volume_fractal_divergence = gap_fractal_5d_momentum * volume_fractal_trend_direction
    intraday_amount_fractal_divergence = intraday_fractal_5d_momentum * amount_fractal_acceleration
    multi_timeframe_fractal_consistency = np.sign(fractal_momentum_5d) * np.sign(fractal_momentum_20d)
    
    # Fractal Divergence Strength Calculation
    bullish_divergence_strength = np.where(bullish_fractal_divergence, 
                                          -fractal_momentum_5d * volume_fractal_5d_return, 0)
    bearish_divergence_strength = np.where(bearish_fractal_divergence, 
                                          fractal_momentum_5d * volume_fractal_5d_return, 0)
    volume_confirmation_strength = np.where(volume_fractal_confirmation,
                                           fractal_momentum_5d * volume_fractal_5d_return, 0)
    
    # Cross-Dimensional Fractal Integration
    gap_intraday_fractal_alignment = gap_fractal_5d_momentum * intraday_fractal_5d_momentum
    liquidity_momentum_fractal_consistency = volume_confirmation_strength * fractal_momentum_acceleration
    
    # Regime-Adaptive Fractal Signal Processing
    volatility_regime_adjustment = np.where(high_fractal_vol_regime, 
                                           short_term_fractal_vol / (medium_term_fractal_vol + 1e-8), 1.0)
    
    # Liquidity Absorption Confidence Weighting
    volume_amount_alignment = np.where(volume_amount_fractal_divergence, 0.5, 1.0)
    volume_magnitude_context = 1 / (data['volume'].rolling(window=5).mean() + 1e-8)
    microstructure_quality = 1 / (liquidity_absorption_memory + 1e-8)
    
    liquidity_confidence_weight = volume_amount_alignment * volume_magnitude_context * microstructure_quality
    
    # Final Fractal Factor Assembly
    fractal_divergence_component = (bullish_divergence_strength + bearish_divergence_strength + 
                                   volume_confirmation_strength)
    
    cross_dimensional_component = (gap_intraday_fractal_alignment + 
                                  liquidity_momentum_fractal_consistency + 
                                  multi_timeframe_fractal_consistency)
    
    range_asymmetry_component = volume_confirmed_asymmetry + range_asymmetry_acceleration
    
    microstructure_component = microstructure_memory_acceleration
    
    # Combine all components
    composite_factor = (fractal_divergence_component + 
                       cross_dimensional_component + 
                       range_asymmetry_component + 
                       microstructure_component)
    
    # Apply regime adjustments and confidence weighting
    final_factor = (composite_factor * volatility_regime_adjustment * 
                   liquidity_confidence_weight)
    
    # Normalize and handle NaN values
    final_factor = final_factor.fillna(0)
    final_factor = final_factor.replace([np.inf, -np.inf], 0)
    
    return final_factor
