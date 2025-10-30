import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate price momentum components
    df = df.copy()
    
    # Short-term and medium-term momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    
    # Volatility regime calculation
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['vol_20d'] = df['daily_range'].rolling(window=20).mean()
    df['vol_60d'] = df['daily_range'].rolling(window=60).mean()
    df['vol_ratio'] = df['vol_20d'] / df['vol_60d']
    
    # Volume trend for confirmation
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_trend'] = df['volume_ma_5'] / df['volume_ma_20'] - 1
    
    # Volatility regime adjustment
    df['vol_regime_adjustment'] = np.where(
        df['vol_ratio'] > 1.2,  # High volatility regime
        0.7,
        np.where(
            df['vol_ratio'] < 0.8,  # Low volatility regime
            1.3,
            1.0  # Normal regime
        )
    )
    
    # Combine momentum with volatility adjustment and volume confirmation
    df['composite_momentum'] = (
        (df['momentum_5d'] * 0.6 + df['momentum_20d'] * 0.4) * 
        df['vol_regime_adjustment'] * 
        (1 + np.tanh(df['volume_trend'] * 2))
    )
    
    # Effective spread proxy using high-low range
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_ma_5'] = df['spread_proxy'].rolling(window=5).mean()
    df['spread_persistence'] = df['spread_proxy'].rolling(window=5).apply(
        lambda x: pd.Series(x).autocorr(lag=1), raw=False
    ).fillna(0)
    
    # Price-volume correlation for liquidity analysis
    df['price_change'] = df['close'].pct_change()
    df['price_volume_corr'] = df['price_change'].rolling(window=10).corr(df['volume'].pct_change()).fillna(0)
    
    # Liquidity momentum signal
    df['liquidity_signal'] = (
        -df['spread_persistence'] *  # Lower persistence suggests better liquidity
        (1 - df['spread_ma_5']) *    # Lower spreads suggest better liquidity
        df['price_volume_corr']      # Positive correlation suggests institutional flow
    )
    
    # Overnight gap analysis
    df['overnight_return'] = df['open'] / df['close'].shift(1) - 1
    df['gap_magnitude'] = abs(df['overnight_return'])
    df['gap_avg_5d'] = df['gap_magnitude'].rolling(window=5).mean()
    df['gap_ratio'] = df['gap_magnitude'] / df['gap_avg_5d']
    
    # Gap persistence tracking
    df['gap_direction'] = np.sign(df['overnight_return'])
    df['consecutive_gaps'] = (
        df['gap_direction'].groupby(
            (df['gap_direction'] != df['gap_direction'].shift(1)).cumsum()
        ).cumcount() + 1
    ) * df['gap_direction']
    
    # Intraday range for mean reversion adjustment
    df['intraday_range'] = (df['high'] - df['low']) / df['open']
    df['range_avg_5d'] = df['intraday_range'].rolling(window=5).mean()
    df['range_ratio'] = df['intraday_range'] / df['range_avg_5d']
    
    # Volume-weighted mean reversion factor
    df['mean_reversion_signal'] = (
        -df['overnight_return'] *  # Reversion direction
        df['gap_ratio'] *          # Magnitude adjustment
        (1 + np.log1p(df['volume'] / df['volume_ma_20'])) *  # Volume weighting
        (1 + df['range_ratio'])    # Intraday range adjustment
    )
    
    # Volume concentration analysis
    df['volume_std_5d'] = df['volume'].rolling(window=5).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma_20']) / df['volume_std_5d'].replace(0, 1)
    df['volume_clustering'] = df['volume_zscore'].rolling(window=5).std()
    
    # Volume-volatility relationship for hidden order detection
    df['vol_volume_corr'] = df['daily_range'].rolling(window=10).corr(df['volume']).fillna(0)
    
    # Order imbalance signal
    df['order_imbalance'] = (
        df['volume_zscore'] *                      # Unusual volume
        (1 - df['volume_clustering']) *           # Less clustering suggests hidden orders
        df['vol_volume_corr'] *                   # Volume-volatility relationship
        df['price_change']                        # Price impact
    )
    
    # ATR-based volatility classification
    df['atr'] = (
        (df['high'] - df['low']).rolling(window=14).mean() / df['close']
    )
    df['atr_regime'] = np.where(
        df['atr'] > df['atr'].rolling(window=60).quantile(0.7),
        'high',
        np.where(
            df['atr'] < df['atr'].rolling(window=60).quantile(0.3),
            'low',
            'normal'
        )
    )
    
    # Adaptive moving averages
    df['vol_weight'] = 1 / (1 + df['atr'] * 10)  # Higher volatility = lower weight
    df['ma_5_vol'] = df['close'].rolling(window=5).apply(
        lambda x: np.average(x, weights=df['vol_weight'].iloc[-len(x):]), raw=False
    )
    df['ma_20_vol'] = df['close'].rolling(window=20).apply(
        lambda x: np.average(x, weights=df['vol_weight'].iloc[-len(x):]), raw=False
    )
    
    # Trend signals with regime adjustment
    df['trend_signal_5d'] = (df['close'] - df['ma_5_vol']) / df['ma_5_vol']
    df['trend_signal_20d'] = (df['close'] - df['ma_20_vol']) / df['ma_20_vol']
    
    # Regime-adjusted trend factor
    df['regime_multiplier'] = np.where(
        df['atr_regime'] == 'high', 0.5,
        np.where(df['atr_regime'] == 'low', 1.5, 1.0)
    )
    
    df['composite_trend'] = (
        (df['trend_signal_5d'] * 0.4 + df['trend_signal_20d'] * 0.6) *
        df['regime_multiplier'] *
        (1 + np.tanh(df['volume_trend']))
    )
    
    # Final composite factor combining all components
    df['final_factor'] = (
        df['composite_momentum'] * 0.25 +
        df['liquidity_signal'] * 0.20 +
        df['mean_reversion_signal'] * 0.25 +
        df['order_imbalance'] * 0.15 +
        df['composite_trend'] * 0.15
    )
    
    return df['final_factor']
