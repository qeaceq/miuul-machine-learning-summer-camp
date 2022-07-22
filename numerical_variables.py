import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

# choosing numerical variables

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

# some of them seems like numerical however we know they are categorical. they include less than 10 value.

# we have cat_cols function from categorical variables.

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# choosing categorical variables from int and float

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# choosing not categorical variables but seems categorical (we should extract these values from categorical variables)

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# we take num_cols which is not in cat_cols

num_cols = [col for col in num_cols if col not in cat_cols]

# we write num_summary function with pre-determined argument: Plot=False


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.005, 0.10, 0.20, 0.50, 0.75, 0.90, 0.95]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:

        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)
    for col in num_cols:
        num_summary(df, col, plot=True)


