import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Hierarchical Intraday Compression-Liquidity Divergence factor
    """
    df = data.copy()
    
    # Multi-Scale Compression Detection
    # Short-Term Range Compression
    df['daily_range'] = df['high'] - df['low']
    df['range_compression'] = df['daily_range'] / df['daily_range'].shift(1)
    
    # Medium-Term Volatility Compression
    df['close_std_5'] = df['close'].rolling(window=5).std()
    df['vol_compression'] = df['close_std_5'] / df['close_std_5'].shift(5)
    
    # Intraday Price Compression
    df['midpoint'] = (df['high'] + df['low']) / 2
    df['intraday_return'] = df['midpoint'] / df['midpoint'].shift(1) - 1
    df['intraday_vol_compression'] = df['intraday_return'].rolling(window=3).std() / \
                                   df['intraday_return'].rolling(window=3).std().shift(3)
    
    # Asymmetric Liquidity Response
    # Directional Volume Asymmetry
    df['close_vs_open'] = df['close'] - df['open']
    df['up_day'] = df['close_vs_open'] > 0
    df['down_day'] = df['close_vs_open'] < 0
    
    # Calculate directional volumes
    df['volume_up'] = df['volume'] * df['up_day']
    df['volume_down'] = df['volume'] * df['down_day']
    
    # Rolling directional volume ratios
    df['volume_up_ratio_3'] = df['volume_up'].rolling(window=3).sum() / \
                             (df['volume_up'].rolling(window=3).sum() + df['volume_down'].rolling(window=3).sum())
    df['volume_down_persistence'] = df['down_day'].rolling(window=5).sum() / 5
    
    # Amount-Weighted Pressure
    df['buying_intensity'] = np.where(df['close'] > df['open'], df['amount'], 0)
    df['selling_absorption'] = np.where(df['close'] < df['open'], df['amount'], 0)
    
    df['buy_pressure_3'] = df['buying_intensity'].rolling(window=3).sum()
    df['sell_pressure_3'] = df['selling_absorption'].rolling(window=3).sum()
    
    # Compression-Liquidity Divergence
    # Net Intraday Pressure
    df['net_intraday_pressure'] = (df['close_vs_open'] / df['open']) * df['volume_up_ratio_3']
    
    # Amount concentration
    df['amount_concentration'] = df['amount'] / df['amount'].rolling(window=10).mean()
    df['net_pressure_weighted'] = df['net_intraday_pressure'] * df['amount_concentration']
    
    # Pressure-Compression Divergence
    # Combined compression score (geometric mean of compression measures)
    compression_measures = ['range_compression', 'vol_compression', 'intraday_vol_compression']
    df['combined_compression'] = df[compression_measures].apply(
        lambda x: np.exp(np.mean(np.log(x.clip(lower=1e-6)))), axis=1
    )
    
    # Raw divergence
    df['raw_divergence'] = df['net_pressure_weighted'] * df['combined_compression']
    
    # Adaptive Factor Construction
    # Compression Depth Weighting
    df['compression_depth'] = 1 / (df['combined_compression'] + 1e-6)
    df['compression_multiplier'] = np.where(
        df['compression_depth'] > df['compression_depth'].rolling(window=20).quantile(0.8),
        2.0,  # Deep compression: aggressive multiplier
        1.0   # Shallow compression: conservative multiplier
    )
    
    # Liquidity Asymmetry Scaling
    df['liquidity_asymmetry'] = (df['buy_pressure_3'] - df['sell_pressure_3']) / \
                               (df['buy_pressure_3'] + df['sell_pressure_3'] + 1e-6)
    
    df['liquidity_strength'] = np.abs(df['liquidity_asymmetry'])
    
    # Final factor construction
    df['compression_liquidity_divergence'] = (
        df['raw_divergence'] * 
        df['compression_multiplier'] * 
        df['liquidity_strength'] * 
        df['amount_concentration']
    )
    
    # Normalize the factor
    factor = df['compression_liquidity_divergence']
    factor = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std()
    
    return factor
