## **DATA**


```python
import numpy as np
import pandas as pd

#read the csv file
data =pd.read_csv( '/content/drive/My Drive/data/IceCreamData.csv')
```

## **DATA SELECTION**


```python
#import the selected data into a Data Frame
#since there are only two fields in the Ice Cream dataset, all of it was stored in the dataframe
data= pd.DataFrame(data)
data.dtypes
```




    Temperature    object
    Revenue        object
    dtype: object



## **PRE PROCESSING**

There are a lot of methods incleaning the data set depending on the requirements. As for this activity, I chose to be as straightforward as possible in cleaning the dataset.


```python
#CLEANING STAGE

#after exploring the dataset, remove unnecessary values
data=data[data.Temperature != 'a' ]
data= data[data.Temperature != 'null']
data=data[data.Revenue != 'b']
```


```python
#get the number of missing data points in each column
data.isnull().sum()
```




    Temperature    10
    Revenue         7
    dtype: int64




```python
#drop the null values
data.dropna(inplace =True)
```


```python
# verify if the null values are removed
data.isnull().sum()
```




    Temperature    0
    Revenue        0
    dtype: int64




```python
#Storing into X the 'Latitude' as np.array
#Storing into y the 'Temp' as np.array
# Vewing the shape of X
# Vewing the shape of y
X = np.array(data[['Temperature']]).reshape(-1,1)
y = np.array(data[['Revenue']]) .reshape(-1,1)
print(X.shape)
print(y.shape)
```

    (481, 1)
    (481, 1)
    


```python
# Another method in removing uneeded values using numpy, please disregard.
# Remove noisy data to avoid error (in this dataset there is a record that has a string value 'a','b','null')
#b = np.array(["a","b","null"])
#X = np.setdiff1d(X,b)
#y = np.setdiff1d(y,b)
```


```python
# to make sure that all values are float
#X = X.astype(np.float)
#y = y.astype(np.float)
X = [float(x) for x in X]
y = [float(z) for z in y]

```


```python
# Show the relationship between Temperature and Revenue
import matplotlib.pyplot as plt
plt.scatter(X,y,color="blue")
plt.title('Temp vs Revenue')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()
```


    
![png](output_13_0.png)
    


## **TRANSFORMATION**

Normalize the dataset using MinMaxScaler.


```python
#preprocess the inequality dataset by normalizing it using the MinMaxScaler.

from sklearn.preprocessing import MinMaxScaler
print ("Activity: Rescale / Normalize dataset: \n")

scaler=MinMaxScaler(feature_range=(0,1))
rescaledSet=scaler.fit_transform(data)
new_data=pd.DataFrame(rescaledSet)
print ("\n Original dataset:")
print(data.head(5))
print ("\n Normalized dataset:")
print(new_data.head(5))
```

    Activity: Rescale / Normalize dataset: 
    
    
     Original dataset:
       Temperature      Revenue
    0  24.56688442  534.7990284
    1  26.00519115  625.1901215
    2  27.79055388  660.6322888
    3  20.59533505  487.7069603
    4  11.50349764  316.2401944
    
     Normalized dataset:
              0         1
    0  0.545931  0.530100
    1  0.577893  0.621404
    2  0.617568  0.657204
    3  0.457674  0.482532
    4  0.255633  0.309334
    

Store the normalized dataset in X and y values to be used in modeling.


```python

X = np.array(new_data[[0]])
y = np.array(new_data[[1]])
```


```python
print(X.shape)
print(y.shape)
```

    (481, 1)
    (481, 1)
    


```python
X = X.reshape(-1,1)
y = y.reshape(-1,1)
```


```python
print(X.shape)
print(y.shape)
```

    (481, 1)
    (481, 1)
    

## **MODELING** *(Data Mining Techniques)*


```python
#splitting the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42, test_size=0.3)
print("Splitting data...")
```

    Splitting data...
    


```python
#import LinearRegression class, instantiate it, and call the fit() method along with our training data
#train the algorithm
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model=model.fit(X_train, y_train)


```


```python
#make predictions on the test data
y_pred=model.predict(X_test)
```

## **EVALUATION/ ANALYSIS**


```python
#compare the actual output values for X_test with the predicted values
df = pd.DataFrame({'Actual Values': y_test.flatten(), 'Predicted Values': y_pred.flatten()})
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual Values</th>
      <th>Predicted Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.587044</td>
      <td>0.588856</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.239179</td>
      <td>0.224514</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.816298</td>
      <td>0.743012</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.638737</td>
      <td>0.624499</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.676477</td>
      <td>0.690458</td>
    </tr>
  </tbody>
</table>
</div>



Comparison of Actual Values vs the Predicted values in graph.


```python
#compare the actual and predicted values using bar graph
df1 = df.head(10)
df1.plot(kind='bar',figsize=(10,7))
#plt.grid(which='actual values', linestyle='-', linewidth='0.5', color='blue')
#plt.grid(which='predicted values', linestyle=':', linewidth='0.5', color='green')
plt.show()
```


    
![png](output_29_0.png)
    



```python
#plot the prediction values
plt.figure(figsize=(15,10))

plt.scatter(X_test, y_test,  color='purple', label= "Plot")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Regression Line")
plt.legend()
plt.grid()
plt.show()
plt.show()
```


    
![png](output_30_0.png)
    



> The figure above shows how the temperature affects the sales of the Ice Cream. It can be clearly seen that as the Temperature rises, the revenue also goes up. The regression line shows the interrelation of the temperature and revenue. Since the plotted values are close to the regression line, it tells us that the model is accurate enough to predict the revenue of the Ice cream based in the temperature.













 **TEST THE MODEL WITH USER INPUT**


```python
while True:
    temp= float(input("Please enter temperature (Celcius) : "))
    prediction = model.predict(([temp], [0]))
    print("Predicted Revenue: $", round(prediction[0,0], 2))
    again = input("Do you want to try again? [Y/N]: ")
    again = again[0].upper()
    if again == 'N' or again == 'n':
        print("\nProgram closed. Thank you!")
        break;
```

    Please enter temperature (Celcius) : 26
    Predicted Revenue: $ 25.42
    Do you want to try again? [Y/N]: y
    Please enter temperature (Celcius) : 15
    Predicted Revenue: $ 14.68
    Do you want to try again? [Y/N]: y
    Please enter temperature (Celcius) : 50
    Predicted Revenue: $ 48.86
    Do you want to try again? [Y/N]: n
    
    Program closed. Thank you!
    
