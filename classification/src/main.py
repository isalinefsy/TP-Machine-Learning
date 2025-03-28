import csv
import pandas as pd
import numpy as np

dataTrain = pd.read_table("./data/train.csv", sep=",")
print(dataTrain)

dataTrain.info()
# No null values

# Check number of rows 
print("Number of rows before deleting: ", len(dataTrain))

# Delete rows where values not between 0 and 1 for date,hour,bc_price,bc_demand,ab_price,ab_demand,transfer
dataTrain = dataTrain[(dataTrain['date'] >= 0) & (dataTrain['date'] <= 1)]
dataTrain = dataTrain[(dataTrain['hour'] >= 0) & (dataTrain['hour'] <= 1)]
dataTrain = dataTrain[(dataTrain['bc_price'] >= 0) & (dataTrain['bc_price'] <= 1)]
dataTrain = dataTrain[(dataTrain['bc_demand'] >= 0) & (dataTrain['bc_demand'] <= 1)]
dataTrain = dataTrain[(dataTrain['ab_price'] >= 0) & (dataTrain['ab_price'] <= 1)]
dataTrain = dataTrain[(dataTrain['ab_demand'] >= 0) & (dataTrain['ab_demand'] <= 1)]
dataTrain = dataTrain[(dataTrain['transfer'] >= 0) & (dataTrain['transfer'] <= 1)]


# delete double rows (identical rows)
dataTrain = dataTrain.drop_duplicates()

#drop identical ids
dataTrain = dataTrain.drop_duplicates(subset=['id'], keep='first')

possiblesValues = ["UP", "DOWN"]

# Check that bc_price_evo column is in the possiblesValues list and drop the rows that are not
dataTrain = dataTrain[dataTrain['bc_price_evo'].isin(possiblesValues)]

# Check finished.

# Sort rows by date 
dataTrain = dataTrain.sort_values(by=['date'])


# Plot the data
import matplotlib.pyplot as plt
# Plot bc_price in blue VS ab_price in red over date (between 15th of May 2015 and the 13th of December 2017 normalized)
plt.plot(dataTrain['date'], dataTrain['bc_price'], color='blue', label='bc_price')
plt.plot(dataTrain['date'], dataTrain['ab_price'], color='red', label='ab_price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Evolution')
plt.legend()
plt.show()

# Plot bc_price_evo in nuage de points with time in x axis
plt.scatter(dataTrain['date'], dataTrain['bc_price_evo'], color='blue', label='bc_price_evo')
plt.xlabel('Date')
plt.ylabel('bc_price_evo')
plt.title('Price Evolution')
plt.legend()
plt.show()


# We think that there is 4 periods of time where we have data




