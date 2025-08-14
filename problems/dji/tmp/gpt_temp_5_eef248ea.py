def heuristics_v2(df):
    def custom_heuristic(row):
        if row.name < 4:
            return 0  # Avoiding NaN or inf values for the first few days
        five_day_avg_close = df.loc[row.name-4:row.name, 'close'].mean()
        return five_day_avg_close / row['volume']
    
    heuristics_matrix = df.apply(custom_heuristic, axis=1)
    return heuristics_matrix
