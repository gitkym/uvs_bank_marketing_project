'''write functions to be utilised later to deal with outliers'''

# write a function to find outliers using IQR method
def iqr(df,col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return lower_bound, upper_bound
# write a function to trim outliers
def trim(df,col):
    lower_bound, upper_bound = iqr(df,col)
    df_trim = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df_trim
# write a function to cap outliers
def cap(df,col):
    lower_bound, upper_bound = iqr(df,col)
    df_cap = df.copy()
    df_cap[col] = df_cap[col].apply(lambda x: lower_bound if x < lower_bound else upper_bound if x > upper_bound else x)
    return df_cap
# write a function to replace outliers with threshold
def thresh(df, col, thresh):
    df_thresh = df.copy()
    df_thresh[col] = df_thresh[col].apply(lambda x: thresh if x > thresh else x)
    return df_thresh
# write a function to replace outliers with median
def replace_median(df,col):
    lower_bound, upper_bound = iqr(df,col)
    df_replace = df.copy()
    median = df_replace[col].median()
    df_replace[col] = df_replace[col].apply(lambda x: median if x < lower_bound or x > upper_bound else x)
    return df_replace
# write a function to replace outliers with mean
def replace_mean(df,col):
    lower_bound, upper_bound = iqr(df,col)
    df_replace = df.copy()
    mean = df_replace[col].mean()
    df_replace[col] = df_replace[col].apply(lambda x: mean if x < lower_bound or x > upper_bound else x)
    return df_replace