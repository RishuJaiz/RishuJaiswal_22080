import pandas as pd
import numpy as np
import statistics

# Load the data
file_path = "C:\\Users\\Rishu Jaiswal\\Downloads\\Lab Session1 Data.xlsx"
df = pd.read_excel(file_path, sheet_name='IRCTC Stock Price')

# Task 1: Calculate the mean and variance of the Price data
price_mean = statistics.mean(df['Price'])
price_variance = statistics.variance(df['Price'])
print("Mean Price:", price_mean)
print("Variance of Price:", price_variance)

# Task 2: Select the price data for all Wednesdays and calculate the sample mean
wednesday_prices = df[df['Day'] == 'Wed']['Price']
if not wednesday_prices.empty:
    wednesday_mean = statistics.mean(wednesday_prices)
    print("Mean Price on Wednesdays:", wednesday_mean)
else:
    print("No data available for Wednesdays.")


# Task 3: Select the price data for the month of April and calculate the sample mean
april_prices = df[df['Month'] == 'Apr']['Price']
april_mean = statistics.mean(april_prices)
print("Mean Price in April:", april_mean)

# Task 4: Find the probability of making a loss over the stock
loss_probability = len(df[df['Chg%'] < 0]) / len(df)
print("Probability of making a loss:", loss_probability)

# Task 5: Calculate the probability of making a profit on Wednesday
wednesday_profit_probability = len(df[(df['Day'] == 'Wed') & (df['Chg%'] > 0)]) / len(df[df['Day'] == 'Wed'])
print("Probability of making a profit on Wednesday:", wednesday_profit_probability)

# Task 6: Calculate the conditional probability of making profit, given that today is Wednesday
conditional_profit_probability = len(df[(df['Day'] == 'Wed') & (df['Chg%'] > 0)]) / len(df)
print("Conditional probability of making profit on Wednesday:", conditional_profit_probability)

# Task 7: Make a scatter plot of Chg% data against the day of the week
import matplotlib.pyplot as plt
plt.scatter(df['Day'], df['Chg%'])
plt.xlabel('Day of the Week')
plt.ylabel('Change Percentage')
plt.title('Change Percentage vs. Day of the Week')
plt.show()
