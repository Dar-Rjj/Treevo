import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def heuristics_v2(data):
    """
    Generate alpha factor combining volatility-adjusted momentum, volume-price divergence,
    and market regime classification with adaptive signal combination.
    """
    df = data.copy()
    
    # Volatility-Adjusted Price Momentum
    # Short-term momentum calculation
    df['ret_5d'] = df['close'] / df['close'].shift(5) - 1
    df['ret_10d'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatility estimation
    df['daily_range_vol'] = (df['high'] - df['low']) / df['close']
    df['avg_range_5d'] = (df['high'].rolling(5).max() - df['low'].rolling(5).min()) / df['close']
    
    # Momentum normalization
    df['norm_mom_5d'] = df['ret_5d'] / df['daily_range_vol'].replace(0, np.nan)
    df['norm_mom_10d'] = df['ret_10d'] / df['avg_range_5d'].replace(0, np.nan)
    
    # Volume-Price Divergence Detection
    # Volume trend analysis
    def calc_volume_slope(volume_series):
        slopes = []
        for i in range(len(volume_series)):
            if i >= 4:
                window = volume_series.iloc[i-4:i+1].values
                X = np.arange(5).reshape(-1, 1)
                model = LinearRegression()
                model.fit(X, window)
                slopes.append(model.coef_[0])
            else:
                slopes.append(np.nan)
        return slopes
    
    df['volume_slope'] = calc_volume_slope(df['volume'])
    df['volume_accel'] = df['volume_slope'] - df['volume_slope'].shift(1)
    
    # Price-volume alignment
    df['direction_agreement'] = np.sign(df['ret_5d']) * np.sign(df['volume_slope'])
    df['strength_ratio'] = np.abs(df['ret_5d']) / (np.abs(df['volume_slope']) + 1e-8)
    
    # Divergence signals
    df['bullish_div'] = ((df['ret_5d'] < 0) & (df['volume_slope'] > 0)).astype(float)
    df['bearish_div'] = ((df['ret_5d'] > 0) & (df['volume_slope'] < 0)).astype(float)
    df['divergence_signal'] = df['bullish_div'] - df['bearish_div']
    
    # Market Regime Classification
    # Volatility regime
    df['high_vol_regime'] = (df['daily_range_vol'] > df['avg_range_5d']).astype(float)
    
    # Trend regime
    df['ma_20d'] = df['close'].rolling(20).mean()
    df['uptrend'] = (df['close'] > df['ma_20d']).astype(float)
    
    # Volume regime
    df['volume_ma_20d'] = df['volume'].rolling(20).mean()
    df['high_volume_regime'] = (df['volume'] > df['volume_ma_20d']).astype(float)
    
    # Adaptive Signal Combination
    # Base momentum signals
    momentum_signal = 0.6 * df['norm_mom_5d'] + 0.4 * df['norm_mom_10d']
    
    # Volume confirmation
    volume_confirmation = np.where(
        (df['direction_agreement'] > 0) & (df['strength_ratio'] > 1), 1.0,
        np.where((df['direction_agreement'] > 0) & (df['strength_ratio'] <= 1), 0.5, 0.0)
    )
    
    # Regime-adaptive weighting
    base_alpha = np.zeros(len(df))
    
    for i in range(len(df)):
        if pd.isna(df['high_vol_regime'].iloc[i]) or pd.isna(df['uptrend'].iloc[i]) or pd.isna(df['high_volume_regime'].iloc[i]):
            base_alpha[i] = np.nan
            continue
            
        # High volatility regimes: emphasize divergence signals
        if df['high_vol_regime'].iloc[i] == 1:
            weight_momentum = 0.3
            weight_divergence = 0.7
        # Low volatility + uptrend: emphasize momentum continuation
        elif df['high_vol_regime'].iloc[i] == 0 and df['uptrend'].iloc[i] == 1:
            weight_momentum = 0.8
            weight_divergence = 0.2
        # Low volatility + downtrend: cautious momentum following
        elif df['high_vol_regime'].iloc[i] == 0 and df['uptrend'].iloc[i] == 0:
            weight_momentum = 0.5
            weight_divergence = 0.5
        else:
            weight_momentum = 0.6
            weight_divergence = 0.4
        
        # High volume regimes: increase volume confirmation weight
        if df['high_volume_regime'].iloc[i] == 1:
            volume_weight = 1.2
        else:
            volume_weight = 1.0
            
        mom_val = momentum_signal.iloc[i] if not pd.isna(momentum_signal.iloc[i]) else 0
        div_val = df['divergence_signal'].iloc[i] if not pd.isna(df['divergence_signal'].iloc[i]) else 0
        vol_conf = volume_confirmation[i] if not pd.isna(volume_confirmation[i]) else 0
        
        base_alpha[i] = (weight_momentum * mom_val + weight_divergence * div_val) * (1 + 0.2 * vol_conf * volume_weight)
    
    # Signal Quality Filtering
    df['base_alpha'] = base_alpha
    
    # Consistency check
    df['signal_persistence'] = df['base_alpha'].rolling(5).apply(
        lambda x: np.sum(np.sign(x.iloc[1:]) == np.sign(x.iloc[0])) if len(x) == 5 else np.nan
    )
    
    df['magnitude_stability'] = df['base_alpha'].rolling(10).std()
    
    # Regime stability
    df['regime_persistence'] = (
        (df['high_vol_regime'].rolling(5).apply(lambda x: np.sum(x.iloc[1:] == x.iloc[0]) if len(x) == 5 else np.nan)) +
        (df['uptrend'].rolling(5).apply(lambda x: np.sum(x.iloc[1:] == x.iloc[0]) if len(x) == 5 else np.nan)) +
        (df['high_volume_regime'].rolling(5).apply(lambda x: np.sum(x.iloc[1:] == x.iloc[0]) if len(x) == 5 else np.nan))
    ) / 3
    
    # Final alpha output with quality filtering
    def apply_quality_filter(row):
        if pd.isna(row['signal_persistence']) or pd.isna(row['magnitude_stability']) or pd.isna(row['regime_persistence']):
            return np.nan
        
        signal_quality = (
            (row['signal_persistence'] / 4) * 0.4 +  # Max persistence is 4
            (1 - min(row['magnitude_stability'] / (np.abs(row['base_alpha']) + 1e-8), 1)) * 0.3 +
            (row['regime_persistence'] / 4) * 0.3    # Max persistence is 4
        )
        
        if signal_quality > 0.7:
            return row['base_alpha'] * 1.2  # High quality boost
        elif signal_quality > 0.4:
            return row['base_alpha']        # Medium quality
        else:
            return row['base_alpha'] * 0.5  # Low quality discount
    
    df['final_alpha'] = df.apply(apply_quality_filter, axis=1)
    
    return df['final_alpha']
