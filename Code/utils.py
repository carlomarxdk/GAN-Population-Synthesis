import numpy as np
import pandas as pd
from itertools import groupby


def numerical_binning(df, cols, q=5):
	r"""
	Bins the numerical variables into q categories using quantile cuts

	Arguments: 
		df: The dataset to be binned
		cols: The columns of df to be binned
		q: The number of cuts
	"""
	for var in cols:
		try:
		    pd.qcut(df[var], q=2)
		    df[var] = pd.cut(df[var], bins=np.quantile(df[var], q=np.linspace(0, 1, num=q)), duplicates='drop', include_lowest=True) # If you want to force it to be of q categories, delete duplicates='drop'
		except:
		    df.drop([var], axis=1, inplace=True)
		    print(var, 'dropped')
		
	return df


def encode_onehot(df, cols=None):
	"""
	Returns the one-hot encoded dataset

	Arguments:
		df: The dataset to be one-hot-encoded
		cols: The variables of df to be one-hot encoded
	"""
	if cols is None:
		cols=list(df)
	df = pd.get_dummies(data=df, columns=cols, sparse=True) ### The sparse=True option might be taken off when using the server
	return df

def back_from_dummies(df):
	r"""
	Returns from one-hot encoding into the original dataset

	Arguments:
		df: the dataset to be returned into a normal one
	"""
	result_series = {}

	# Find dummy columns and build pairs (category, category_value)
	dummmy_tuples = [(col.split("_")[0],col) for col in df.columns if "_" in col]

	# Find non-dummy columns that do not have a _
	non_dummy_cols = [col for col in df.columns if "_" not in col]

	# For each category column group use idxmax to find the value.
	for dummy, cols in groupby(dummmy_tuples, lambda item: item[0]):

		#Select columns for each category
		dummy_df = df[[col[1] for col in cols]]

		# Find max value among columns
		max_columns = dummy_df.idxmax(axis=1)

		# Remove category_ prefix
		result_series[dummy] = max_columns.apply(lambda item: item.split("_")[1])

	# Copy non-dummy columns over.
	for col in non_dummy_cols:
		result_series[col] = df[col]

	# Return dataframe of the resulting series
	return pd.DataFrame(result_series)