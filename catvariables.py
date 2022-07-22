import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.set_option("display.max.columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
# we can see all information about data set with typing -df.info()- on console

# choosing all categorical variables

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# choosing categorical variables from int and float

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# choosing not categorical variables but seems categorical (we should extract these values from categorical variables)

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

# we've chosen all categorical variables.

# defining summary function with plot feature.


def cat_summary(dataframe, col_name, plot: False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name])
        plt.show(block=True)


# generalizing our function for all categorical variables

for cal in cat_cols:
    cat_summary(df, cal, plot=True)

# we add "boolean" type and convert to integer

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)











