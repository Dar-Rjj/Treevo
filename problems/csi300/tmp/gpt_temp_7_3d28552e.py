import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Velocity Acceleration with Liquidity Dampening
    # Calculate True Range Velocity
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    # True Range components
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = np.abs(df['high'] - df['prev_close'])
    df['tr3'] = np.abs(df['low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['tr_velocity'] = df['true_range']  # Daily velocity (divided by 1 day)
    
    # Close price velocity (first derivative)
    df['close_velocity'] = df['close'] - df['prev_close']
    
    # Combined velocity measure
    df['price_velocity'] = (df['tr_velocity'] + np.abs(df['close_velocity'])) / 2
    
    # Acceleration (second derivative)
    df['velocity_change'] = df['price_velocity'] - df['price_velocity'].shift(1)
    df['acceleration'] = df['velocity_change']
    
    # Liquidity measurement
    df['avg_trade_size'] = df['amount'] / df['volume']
    df['avg_trade_size'] = df['avg_trade_size'].replace([np.inf, -np.inf], np.nan)
    
    # Rolling liquidity percentiles (21-day window)
    df['liquidity_percentile'] = df['avg_trade_size'].rolling(window=21, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Dampening factor based on liquidity
    df['liquidity_dampening'] = 1 / (1 + np.exp(-df['liquidity_percentile']))
    
    # Final factor: acceleration scaled by liquidity dampening
    df['acceleration_factor'] = df['acceleration'] * df['liquidity_dampening']
    
    # Volatility-Constrained Range Breakout
    # Daily range percentage
    df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Range expansion/contraction
    df['range_ratio'] = df['daily_range_pct'] / df['daily_range_pct'].shift(1)
    
    # Rolling volatility (10-day historical)
    df['price_volatility'] = df['close'].shift(1).rolling(window=10, min_periods=5).std()
    
    # Dynamic breakout thresholds
    df['volatility_band_upper'] = df['price_volatility'] * 2
    df['volatility_band_lower'] = df['price_volatility'] * 0.5
    
    # Breakout detection
    df['range_breakout'] = ((df['daily_range_pct'] > df['volatility_band_upper']) | 
                           (df['daily_range_pct'] < df['volatility_band_lower'])).astype(int)
    
    # Close price momentum
    df['close_momentum'] = df['close'] / df['close'].shift(5) - 1
    
    # Combined breakout signal
    df['breakout_signal'] = df['range_breakout'] * df['close_momentum']
    
    # Microstructure Informed Reversal
    # Trade size analysis
    df['trade_size_skew'] = df['avg_trade_size'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.median()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Price elasticity (market impact)
    df['price_response'] = (df['close'] - df['prev_close']) / df['volume'].replace(0, np.nan)
    df['market_impact'] = df['price_response'].rolling(window=10, min_periods=5).mean()
    
    # Overreaction detection
    df['price_move_vs_volume'] = np.abs(df['close_velocity']) / (df['volume'] + 1e-6)
    df['overreaction_score'] = df['price_move_vs_volume'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Reversal composite
    df['reversal_signal'] = df['trade_size_skew'] * df['overreaction_score']
    
    # Gap-Fill Probability
    df['price_gap'] = (df['open'] - df['prev_close']) / df['prev_close']
    df['gap_magnitude'] = np.abs(df['price_gap'])
    
    # Volume analysis for gaps
    df['volume_surge_ratio'] = df['volume'] / df['volume'].rolling(window=10, min_periods=5).mean()
    
    # Gap fill probability (simplified)
    df['fill_probability'] = 1 / (1 + df['gap_magnitude'] * (2 - df['volume_surge_ratio']))
    
    # Momentum Fracture Detection
    # Momentum calculation
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Momentum fracture points
    df['momentum_change'] = df['momentum_5d'] - df['momentum_5d'].shift(1)
    df['momentum_fracture'] = np.abs(df['momentum_change']) / (df['momentum_5d'].rolling(window=10, min_periods=5).std() + 1e-6)
    
    # Volume confirmation
    df['volume_fracture_ratio'] = df['volume'] / df['volume'].shift(1)
    
    # Composite fracture index
    df['fracture_signal'] = df['momentum_fracture'] * df['volume_fracture_ratio']
    
    # Final combined factor (weighted combination)
    factors = ['acceleration_factor', 'breakout_signal', 'reversal_signal', 'fill_probability', 'fracture_signal']
    weights = [0.25, 0.25, 0.2, 0.15, 0.15]
    
    # Clean and combine factors
    combined_factor = pd.Series(0, index=df.index)
    for factor, weight in zip(factors, weights):
        clean_factor = df[factor].fillna(0).replace([np.inf, -np.inf], 0)
        combined_factor += weight * clean_factor
    
    # Remove intermediate columns
    cols_to_drop = ['prev_high', 'prev_low', 'prev_close', 'tr1', 'tr2', 'tr3', 
                   'true_range', 'tr_velocity', 'close_velocity', 'price_velocity',
                   'velocity_change', 'acceleration', 'avg_trade_size', 'liquidity_percentile',
                   'liquidity_dampening', 'daily_range_pct', 'range_ratio', 'price_volatility',
                   'volatility_band_upper', 'volatility_band_lower', 'range_breakout',
                   'close_momentum', 'trade_size_skew', 'price_response', 'market_impact',
                   'price_move_vs_volume', 'overreaction_score', 'price_gap', 'gap_magnitude',
                   'volume_surge_ratio', 'momentum_5d', 'momentum_10d', 'momentum_change',
                   'momentum_fracture', 'volume_fracture_ratio'] + factors
    
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    return combined_factor
