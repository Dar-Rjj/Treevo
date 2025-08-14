import pandas as pd

def heuristics_v2(df):
    ema_spans = [5, 10, 20]  # Example EMA spans
    emas = {span: df.apply(lambda x: x.ewm(span=span).mean(), axis=0) for span in ema_spans}
    avg_ema = pd.concat(emas.values()).groupby(level=1).mean()
    log_ratio = (df['close'] / avg_ema).apply(np.log)
    heuristics_matrix = log_ratio.sum(axis=1)
    return heuristics_matrix
