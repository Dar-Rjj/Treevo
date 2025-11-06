import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Price-Volume Divergence with Fractal Momentum Structure
    """
    df = df.copy()
    
    # Calculate Fractal Momentum Components
    def fractal_momentum_3day(high, low, close):
        # Directional persistence score
        close_returns = close.pct_change()
        up_count = (close_returns > 0).rolling(3).sum()
        down_count = (close_returns < 0).rolling(3).sum()
        persistence = (up_count - down_count) / 3
        
        # Magnitude consistency
        magnitude_std = close_returns.rolling(3).std()
        magnitude_mean = close_returns.rolling(3).mean().abs()
        consistency = magnitude_mean / (magnitude_std + 1e-8)
        
        # Fractal efficiency ratio
        price_change = (close - close.shift(3)).abs()
        total_oscillation = (high.rolling(3).max() - low.rolling(3).min())
        efficiency = price_change / (total_oscillation + 1e-8)
        
        return persistence + consistency + efficiency
    
    def fractal_momentum_8day(high, low, close):
        # Price path complexity
        close_changes = close.diff()
        directional_changes = ((close_changes * close_changes.shift(1)) < 0).rolling(8).sum()
        complexity = 1 - (directional_changes / 7)
        
        # Price excursion efficiency
        net_move = (close - close.shift(8)).abs()
        total_range = (high.rolling(8).max() - low.rolling(8).min())
        excursion_eff = net_move / (total_range + 1e-8)
        
        # Momentum persistence
        up_moves = (close_changes > 0).rolling(8).sum()
        down_moves = (close_changes < 0).rolling(8).sum()
        momentum_strength = (up_moves - down_moves) / 8
        
        volatility = close.pct_change().rolling(8).std()
        adj_momentum = momentum_strength / (volatility + 1e-8)
        
        return complexity + excursion_eff + adj_momentum
    
    def fractal_momentum_21day(high, low, close):
        # Trend quality - price path smoothness
        returns = close.pct_change()
        smoothness = 1 - (returns.rolling(21).std() / returns.rolling(21).mean().abs().replace(0, 1e-8))
        
        # Trend acceleration
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(21).mean()
        acceleration = (ma_short - ma_short.shift(5)) - (ma_long - ma_long.shift(5))
        
        # Fractal dimension approximation
        price_range = high.rolling(21).max() - low.rolling(21).min()
        total_movement = close.diff().abs().rolling(21).sum()
        fractal_dim = total_movement / (price_range + 1e-8)
        
        return smoothness + acceleration + fractal_dim
    
    # Calculate Hierarchical Volume-Price Divergence
    def volume_distribution_patterns(volume, close):
        # Volume concentration metrics
        volume_skew = volume.rolling(5).skew()
        volume_clustering = volume.rolling(5).std() / (volume.rolling(5).mean() + 1e-8)
        
        # Volume persistence
        volume_ma = volume.rolling(5).mean()
        volume_above_ma = (volume > volume_ma).rolling(10).mean()
        volume_trend = volume.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        return volume_skew + volume_clustering + volume_above_ma + volume_trend
    
    def price_volume_synchronization(close, volume):
        # Directional alignment
        price_dir = np.sign(close.diff())
        volume_dir = np.sign(volume.diff())
        alignment = (price_dir == volume_dir).rolling(5).mean()
        
        # Magnitude coordination
        price_move = close.pct_change().abs()
        volume_move = volume.pct_change().abs()
        magnitude_corr = price_move.rolling(5).corr(volume_move)
        
        # Divergence severity
        price_trend = close.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        volume_trend = volume.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        divergence = np.abs(price_trend * volume_trend) * np.sign(price_trend * volume_trend)
        
        return alignment + magnitude_corr - divergence
    
    def multi_scale_divergence(close, volume, high, low):
        # Micro-scale (3-day)
        micro_price = close.pct_change(3)
        micro_volume = volume.pct_change(3)
        micro_div = micro_price * micro_volume
        
        # Meso-scale (8-day)
        meso_price = close.rolling(8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        meso_volume = volume.rolling(8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        meso_div = meso_price * meso_volume
        
        # Macro-scale (21-day)
        macro_price = close.rolling(21).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        macro_volume = volume.rolling(21).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        macro_div = macro_price * macro_volume
        
        return micro_div + meso_div + macro_div
    
    # Synthesize Fractal Divergence Alpha
    # Fractal momentum layers
    micro_momentum = fractal_momentum_3day(df['high'], df['low'], df['close'])
    meso_momentum = fractal_momentum_8day(df['high'], df['low'], df['close'])
    macro_momentum = fractal_momentum_21day(df['high'], df['low'], df['close'])
    
    # Weight by reliability (volatility-adjusted)
    micro_weight = 1 / (df['close'].pct_change().rolling(3).std() + 1e-8)
    meso_weight = 1 / (df['close'].pct_change().rolling(8).std() + 1e-8)
    macro_weight = 1 / (df['close'].pct_change().rolling(21).std() + 1e-8)
    
    combined_momentum = (micro_momentum * micro_weight + 
                        meso_momentum * meso_weight + 
                        macro_momentum * macro_weight) / (micro_weight + meso_weight + macro_weight + 1e-8)
    
    # Price-volume divergence signals
    volume_patterns = volume_distribution_patterns(df['volume'], df['close'])
    price_volume_sync = price_volume_synchronization(df['close'], df['volume'])
    multi_divergence = multi_scale_divergence(df['close'], df['volume'], df['high'], df['low'])
    
    # Composite alpha factor
    divergence_confirmation = volume_patterns * price_volume_sync * multi_divergence
    alpha_factor = combined_momentum * divergence_confirmation
    
    # Apply Adaptive Signal Refinement
    # Pattern quality assessment
    momentum_consistency = (micro_momentum.rolling(5).std() + 
                          meso_momentum.rolling(5).std() + 
                          macro_momentum.rolling(5).std()) / 3
    
    divergence_consistency = (volume_patterns.rolling(5).std() + 
                            price_volume_sync.rolling(5).std() + 
                            multi_divergence.rolling(5).std()) / 3
    
    pattern_quality = 1 / (momentum_consistency + divergence_consistency + 1e-8)
    
    # Signal confidence calibration
    volume_conviction = df['volume'].rolling(10).mean() / (df['volume'].rolling(30).mean() + 1e-8)
    price_validation = (df['close'].rolling(5).mean() > df['close'].rolling(20).mean()).astype(float)
    
    final_alpha = alpha_factor * pattern_quality * volume_conviction * price_validation
    
    return final_alpha
