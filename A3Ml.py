import pandas as pd

file_path = 'C:/Users/Rishu Jaiswal/Downloads/Lab Session1 Data.xlsx'
df = pd.read_excel(file_path)
purchase_data = pd.read_excel(file_path)
purchase_data['Customer_Type'] = purchase_data['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

features = ['Customer', 'Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']  

# Split data into features and target variable
X = purchase_data[features]
y = purchase_data['Customer_Type']

print(y)

