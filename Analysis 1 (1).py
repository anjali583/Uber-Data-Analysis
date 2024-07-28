#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_kmo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data=pd.read_csv("uber.csv")
data


# In[3]:


#converting pickup_datetime to datetime
data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])


# In[4]:


# Check unique values in passenger_count
print("Unique values in passenger_count:", data['passenger_count'].unique())

# Check for any abnormal values in longitude and latitude
print("Longitude range:", data[['pickup_longitude', 'dropoff_longitude']].describe())
print("Latitude range:", data[['pickup_latitude', 'dropoff_latitude']].describe())


# In[5]:


# Filter out outliers in longitude and latitude
data = data[(data['pickup_longitude'] > -80) & (data['pickup_longitude'] < -70) &
            (data['dropoff_longitude'] > -80) & (data['dropoff_longitude'] < -70) &
            (data['pickup_latitude'] > 35) & (data['pickup_latitude'] < 45) &
            (data['dropoff_latitude'] > 35) & (data['dropoff_latitude'] < 45)]


# In[6]:


summary=data.describe()
summary


# # Visualization

# In[7]:


#making histogram for the fare amount to see the distribution of fare_amount
plt.figure(figsize=(12, 8))
sns.histplot(data['fare_amount'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Fare Amounts', fontsize=16)
plt.xlabel('Fare Amount ($)', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[8]:


#scatter plot of the pickup location
plt.figure(figsize=(12, 8))
sns.scatterplot(x='pickup_longitude', y='pickup_latitude', hue='passenger_count', palette='viridis', data=data, s=50, alpha=0.6)
plt.title('Pickup Locations', fontsize=16)
plt.xlabel('Longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.legend(title='Passenger Count')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[9]:


#calculatinng the distance of trip
data['trip_distance'] = np.sqrt((data['dropoff_longitude'] - data['pickup_longitude'])**2 + 
                                (data['dropoff_latitude'] - data['pickup_latitude'])**2)


# In[10]:


data=data.dropna()
print(data.isnull().sum())


# # regression

# In[11]:


# Regression model
x = data[['trip_distance', 'passenger_count']]
y = data['fare_amount']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[14]:


model = LinearRegression().fit(x_train, y_train)


# In[15]:


data['predicted_fare'] = model.predict(x)


# In[16]:


# Plot actual vs predicted fare
plt.figure(figsize=(12, 8))
sns.scatterplot(x='fare_amount', y='predicted_fare', data=data, alpha=0.6, s=50)
plt.plot([data['fare_amount'].min(), data['fare_amount'].max()], [data['fare_amount'].min(), data['fare_amount'].max()], 'r--', linewidth=2)
plt.title('Actual vs Predicted Fare', fontsize=16)
plt.xlabel('Actual Fare ($)', fontsize=14)
plt.ylabel('Predicted Fare ($)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# In[17]:


data.head()


# In[18]:


import pandas as pd

data.to_csv("C:\\Users\\hp\\Desktop\\Regression\\.CSV FILES\\data.csv", index=False)



# # FACTOR ANALYSIS
# 

# In[19]:


import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_kmo
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bartlett, shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


# In[20]:


numerical_data = data[['fare_amount', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']]


# In[21]:


scaler = StandardScaler()
scaled_data = scaler.fit_transform(numerical_data)


# In[22]:


# Perform KMO test
kmo_all, kmo_model = calculate_kmo(scaled_data)
print(f"KMO Model: {kmo_model}")


# In[23]:


# Perform Bartlett's test
from scipy.stats import bartlett
chi_square_value, p_value = bartlett(*scaled_data.T)
print(f"Bartlett's Test: chi-square value = {chi_square_value}, p-value = {p_value}")


# In[24]:


# Perform factor analysis
fa = FactorAnalyzer(rotation=None)
fa.fit(scaled_data)


# In[25]:


# Eigenvalue checking
ev, v = fa.get_eigenvalues()


# In[26]:


# Scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(ev) + 1), ev, 'o-', markersize=8)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid(True)
plt.show()


# In[27]:


n_factors = 4


# In[28]:


# Perform factor analysis with rotation
fa = FactorAnalyzer(n_factors=n_factors, rotation='equamax')
fa.fit(scaled_data)


# In[29]:


# Get factor loadings
loadings = fa.loadings_
print(pd.DataFrame(loadings, index=numerical_data.columns, columns=[f'Factor {i+1}' for i in range(n_factors)]))


# In[30]:


# Compute factor scores
factor_scores = fa.transform(scaled_data)


# In[31]:


# Assign proper names to factors
factor_names = ['Fare & Longitude', 'Latitude', 'Longitude at Dropoff', 'Passenger Count']
for i, name in enumerate(factor_names):
    data[name] = factor_scores[:, i]


# In[32]:


# Step 2: Perform regression analysis
# Define independent variables (factors) and dependent variable (fare_amount)
X = data[factor_names]
y = data['fare_amount']


# In[33]:


import statsmodels.api as sm


# In[34]:


# Add constant term for intercept
X = sm.add_constant(X)


# In[35]:


# Fit the regression model
model = sm.OLS(y, X).fit()


# In[36]:


# Print regression summary
print(model.summary())


# In[ ]:


#Linearity
plt.figure(figsize=(10, 6))
sns.regplot(x=model.fittedvalues, y=model.resid, lowess=True, line_kws={'color': 'red'})
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')
plt.show()


# In[ ]:


# Independence (Durbin-Watson test)
durbin_watson_stat = sm.stats.stattools.durbin_watson(model.resid)
print(f"Durbin-Watson statistic: {durbin_watson_stat}")


# In[ ]:


# Homoscedasticity
plt.figure(figsize=(10, 6))
sns.scatterplot(x=model.fittedvalues, y=np.sqrt(np.abs(model.resid)))
plt.xlabel('Fitted values')
plt.ylabel('Square root of absolute residuals')
plt.title('Scale-Location Plot')
plt.axhline(y=np.mean(np.sqrt(np.abs(model.resid))), color='red')
plt.show()


# In[ ]:


# Normality of residuals
plt.figure(figsize=(10, 6))
sns.histplot(model.resid, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


# Q-Q plot
sm.qqplot(model.resid, line='45', fit=True)
plt.title('Q-Q Plot')
plt.show()


# In[ ]:


# Shapiro-Wilk test
shapiro_test = shapiro(model.resid)
print(f"Shapiro-Wilk test: Statistic = {shapiro_test.statistic}, p-value = {shapiro_test.pvalue}")


# In[ ]:


# Multicollinearity
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# In[ ]:


Predict fare amounts using the regression model
data['predicted_fare'] = model.predict(X)


# In[ ]:


# Plot actual vs predicted fares
plt.figure(figsize=(10, 6))
sns.scatterplot(x='fare_amount', y='predicted_fare', data=data)
plt.plot([data['fare_amount'].min(), data['fare_amount'].max()],
         [data['fare_amount'].min(), data['fare_amount'].max()],
         color='red', lw=2)
plt.xlabel('Actual Fare')
plt.ylabel('Predicted Fare')
plt.title('Actual vs Predicted Fare Amount')
plt.show()


# In[ ]:


# Evaluate the regression model
mse = mean_squared_error(y, data['predicted_fare'])
r2 = r2_score(y, data['predicted_fare'])

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")



# In[ ]:




