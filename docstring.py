# firstly we've changed our docstring format from:
# settings > search bar > type docstring > python ıntegrated tools > docstring format > change as "NumPy"

# firstly we need to bring our previous functions

# CATEGORICAL VARIABLES

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


# NUMERICAL VARIABLES

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

# ADDING DOCSTRING

# firstly we need to clarify our function's aim.
# secondly we give data information of parameters.
# then we must determine what parameters do.
# what kind of statements we will see in the output. we determine these
# in returns.


def grab_col_names(dataframe, cat_th=10, car_th=20):

    """
    This function give numerical, categorical and categorical but cardinal variables in the data set.

    Parameters
    ----------
    dataframe : dataframe
        dataframe that we want to bring our variable's names.
    cat_th : int,float
        it gives numerical but categorical variables.
    car_th : int,float
        it gives categorical but cardinal(for example "names". There
         are several of them, they don't represent any meaningful data)
         variables.

    Returns
    -------
    cat_cols: list
        categorical variables list.
    num_cols : list
        numerical variables list.
    cat_but_car : list
        unnecessary values that seems categorical.

    Notes
    -------
    cat_cols + num_cols + cat_but_car = total variable number
    num_but_cat in cat_cols.


    """

