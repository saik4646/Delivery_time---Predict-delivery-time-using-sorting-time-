problem_statements:

1)Q .Delivery_time -> Predict delivery time using sorting time 

Ans:delivery_time_csv (file)
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Load the dataset
file_path = ('F:\delivery_time.csv')
data = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(data.head())
# Visualize the data with labeled points
plt.scatter(data['Sorting Time'], data['Delivery Time'], label='Data Points')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Sorting Time vs Delivery Time')
plt.legend()
plt.show()
# Prepare the data for modeling
X = data[['Sorting Time']]
y = data[['Delivery Time']]
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Print the model coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_[0]}')
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
# Plot the regression line
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Linear Regression Model')
plt.legend()
plt.show()

                     ** OUTPUT**
  
 1) ![image](https://github.com/saik4646/Delivery_time---Predict-delivery-time-using-sorting-time-/assets/150954390/65ee27cf-3ca1-477d-a09b-e51df75b7de8) 

2) Intercept: [4.68229796]
 Coefficient: [2.02424455]
3) Mean Squared Error: 14.046738956635016
4) ![image](https://github.com/saik4646/Delivery_time---Predict-delivery-time-using-sorting-time-/assets/150954390/a4ae4b40-ac7b-4101-9a7c-a3e0af3efb74)


2)Q. Salary_hike -> Build a prediction model for Salary_hike
Ans:  Salary_data.csv(file).
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load the dataset
file_path = r'F:\Salary_data.csv'
data = pd.read_csv(file_path)
# Display the first few rows of the dataset
print(data.head())
# EDA and Data Visualization
plt.scatter(data['YearsExperience'], data['Salary'])
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Scatter Plot of Years of Experience vs. Salary')
plt.show()
# Split the data into training and testing sets
X = data[['YearsExperience']]
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
# Plot the regression line
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Simple Linear Regression Model')
plt.legend()
plt.show()
                                               OUTPUT   
      1) ![image](https://github.com/saik4646/Delivery_time---Predict-delivery-time-using-sorting-time-/assets/150954390/93ed69d6-0957-4847-8b39-e07cf583f121)
                       
     2)  Mean Squared Error: 49830096.85590839
          R-squared: 0.9024461774180497   
          
      3) ![image](https://github.com/saik4646/Delivery_time---Predict-delivery-time-using-sorting-time-/assets/150954390/232b4b06-2be0-475c-9bc5-0a3bcf08d55c)

          
