# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv(r"C:\DataScience\MyProjects\HousePricePrediction\house_price_dataset.csv")
dataframe = pd.DataFrame(dataset)
print(dataframe.columns)

# dropping unnecessary columns
dataframe.drop(dataframe.columns[0],axis=1,inplace=True)

# renaming the columns
df = dataframe.rename(columns={'squareMeters':'sq.mtrs','numberOfRooms':'no.of_rooms','hasYard':'yard','hasPool':'pool','cityCode':'city_code','cityPartRange':'city_part_range','numPrevOwners':'no.of_prev_owners','isNewBuilt':'new_built','hasStormProtector':'storm_protector','hasStorageRoom':'store_room','hasGuestRoom':'guest_room'})
print(df.columns)

# To obtain the datatypes of each columns
print(df.dtypes)

# To generate summary statistics for the numerical columns in a Data.
print(df.describe())

# Checking for null values.
print(df.isnull().sum())

# Checking for duplicate values.
print(df.duplicated().sum())

# Correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,linewidths=0.5)
plt.title('Heatmap')
plt.show()

# Extracting Independent and dependent Variable
x = df.drop(columns=['price'])
y = df['price']

# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
stc_x= StandardScaler()
x_train= stc_x.fit_transform(x_train)
x_test= stc_x.transform(x_test)

# Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)


# Predicting the test set result
y_pred = regressor.predict(x_test)
df2=pd.DataFrame({"Actual Y-Data":y_test,"Predicted Data":y_pred})
print("Prediction Result")
print(df2.to_string())

# Evaluating the Algorithm
from sklearn.metrics import mean_squared_error ,r2_score
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

# Predicting the accuracy score
r2 = r2_score(y_test, y_pred)
print("r2 score is ",r2*100,"%")
