import pandas as pd
#to upload csv file

data_sales= pd.read_csv("grocery_sales.csv")

#fill the missing entries

avg_sales = data_sales["sales"].mean()
data_sales["sales"].fillna(value = avg_sales, inplace = True)

#sum sales
sales_summary = data_sales.groupby("transaction_date")["sales"].sum()

#plot summary
sales_summary.plot()
