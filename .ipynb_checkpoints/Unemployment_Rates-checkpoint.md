# An Empirical Comparison of Machine Learning Models for Early Prediction of Unemployment Rates Across Countries




## Introduction 


### Background 
Unemployment rates serve as a critical economic indicator, reflecting the health of labor markets and influencing socio-economic stability worldwide. High unemployment can lead to reduced consumer spending, increased poverty, and social unrest, while low unemployment rates often signal economic prosperity. The ability to predict unemployment rates early is vital for policymakers to design proactive interventions, such as job creation programs and economic stimulus packages. This project leverages a comprehensive dataset of unemployment rates from 1991 to 2024 across various countries to analyze historical trends and develop machine learning models for early prediction. By comparing different models, this study aims to identify the most effective approach for forecasting unemployment, contributing to informed economic decision-making globally.


### Research Problem

Early prediction of unemployment rates is challenging due to the complex interplay of economic, social, and political factors that vary across countries. Historical unemployment data often contains missing values, outliers, and inconsistencies, complicating the application of machine learning models. Furthermore, external shocks like the 2008 financial crisis or the 2020 COVID-19 pandemic introduce volatility, making it difficult to capture consistent patterns. This project addresses the problem of developing accurate machine learning models to predict unemployment rates using historical data from 1991 to 2024. The challenge lies in handling incomplete datasets, mitigating the impact of outliers, and selecting models that can generalize across diverse economic contexts for reliable early predictions.

### Objectives

1.To assess the temporal and regional patterns of unemployment rates across 266 countries from 1991 to 2024 using descriptive statistics and visualizations.
>
2.To develop three machine learning models (linear regression,Random Forest and XGBoost) for predicting unemployment rates one year ahead across countries.
>
3.To compare the performance of linear regression,Random Forest and XGBoost models using Mean Squared Error (MSE) and R-squared metrics to identify the most accurate model for early unemployment rate prediction.



#### METHODOLOGY RESULTS AND DISCUSSION 

The dataset, sourced from the World Bank’s World Development Indicators (SL.UEM.TOTL.ZS), provides unemployment rates (% of total labor force) for 266 entities, including 193 countries (e.g., Afghanistan, Angola) and 73 regional aggregates (e.g., Africa Eastern and Southern), spanning January 1991 to December 2024. Collected annually by the World Bank through national labor force surveys and administrative records, it was downloaded as a CSV file in July 2025 from the World Bank’s open data portal. The dataset captures diverse economic conditions, such as the 2008 global financial crisis, which elevated unemployment in developed economies, and the 2020 COVID-19 pandemic, which triggered widespread job losses globally. Structured with 36 columns, including metadata (Country Name, Country Code, Indicator Name, Indicator Code) and yearly rates, it is well-suited for time-series analysis and machine learning to predict unemployment rates years ahead. Its extensive temporal (34 years) and geographical coverage enables robust trend analysis and cross-country comparisons, critical for the research’s objective of comparing predictive models like Random Forest and LSTM. However, issues such as missing data for smaller entities (e.g., Aruba, Andorra) due to inconsistent reporting and outliers from economic disruptions require thorough cleaning to support reliable modeling. The dataset’s standardized format, aligned with ILO(International Labour Organization) definitions, enhances its reliability, though variations in national data quality may introduce minor inconsistencies.


```python
%pip install scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
%pip install xgboost


```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: scikit-learn in c:\programdata\anaconda3\lib\site-packages (1.5.1)
    Requirement already satisfied: numpy>=1.19.5 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.26.4)
    Requirement already satisfied: scipy>=1.6.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.13.1)
    Requirement already satisfied: joblib>=1.2.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=3.1.0 in c:\programdata\anaconda3\lib\site-packages (from scikit-learn) (3.5.0)
    Note: you may need to restart the kernel to use updated packages.
    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: xgboost in c:\users\light house\appdata\roaming\python\python312\site-packages (3.0.2)
    Requirement already satisfied: numpy in c:\programdata\anaconda3\lib\site-packages (from xgboost) (1.26.4)
    Requirement already satisfied: scipy in c:\programdata\anaconda3\lib\site-packages (from xgboost) (1.13.1)
    Note: you may need to restart the kernel to use updated packages.
    

## Step 1: Loading the Dataset

The dataset contains unemployment rates (% of total labor force) for 266 countries and regions from 1991 to 2024, sourced from an international database, likely the World Bank or International Labour Organization, given the indicator code (SL.UEM.TOTL.ZS). It includes entities such as individual countries and regional aggregates. The data was collected annually, capturing unemployment trends under varying economic conditions.


```python
df_data = pd.read_csv("Unemployment_Rate_Dataset.csv") # Load dataset
df_data.head(20)
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Indicator Name</th>
      <th>Indicator Code</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>...</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
      <th>2024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>ABW</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Africa Eastern and Southern</td>
      <td>AFE</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>8.179629</td>
      <td>8.270724</td>
      <td>8.266327</td>
      <td>8.138291</td>
      <td>7.908446</td>
      <td>7.823908</td>
      <td>...</td>
      <td>7.036357</td>
      <td>7.194666</td>
      <td>7.346331</td>
      <td>7.360513</td>
      <td>7.584419</td>
      <td>8.191395</td>
      <td>8.577385</td>
      <td>7.985202</td>
      <td>7.806365</td>
      <td>7.772654</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>AFG</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>8.070000</td>
      <td>8.011000</td>
      <td>7.888000</td>
      <td>7.822000</td>
      <td>7.817000</td>
      <td>7.867000</td>
      <td>...</td>
      <td>9.052000</td>
      <td>10.133000</td>
      <td>11.184000</td>
      <td>11.196000</td>
      <td>11.185000</td>
      <td>11.710000</td>
      <td>11.994000</td>
      <td>14.100000</td>
      <td>13.991000</td>
      <td>13.295000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Africa Western and Central</td>
      <td>AFW</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>4.158680</td>
      <td>4.251102</td>
      <td>4.369805</td>
      <td>4.393781</td>
      <td>4.399749</td>
      <td>4.340691</td>
      <td>...</td>
      <td>4.164467</td>
      <td>4.157574</td>
      <td>4.274196</td>
      <td>4.323631</td>
      <td>4.395271</td>
      <td>4.852393</td>
      <td>4.736732</td>
      <td>3.658573</td>
      <td>3.277245</td>
      <td>3.218313</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>AGO</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>16.855000</td>
      <td>16.978000</td>
      <td>17.399000</td>
      <td>17.400000</td>
      <td>16.987000</td>
      <td>16.275000</td>
      <td>...</td>
      <td>16.490000</td>
      <td>16.575000</td>
      <td>16.610000</td>
      <td>16.594000</td>
      <td>16.497000</td>
      <td>16.690000</td>
      <td>15.799000</td>
      <td>14.602000</td>
      <td>14.537000</td>
      <td>14.464000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Albania</td>
      <td>ALB</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>10.304000</td>
      <td>30.007000</td>
      <td>25.251000</td>
      <td>20.835000</td>
      <td>14.607000</td>
      <td>13.928000</td>
      <td>...</td>
      <td>17.193000</td>
      <td>15.418000</td>
      <td>13.616000</td>
      <td>12.304000</td>
      <td>11.466000</td>
      <td>11.690000</td>
      <td>11.474000</td>
      <td>10.137000</td>
      <td>10.108000</td>
      <td>10.250000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Andorra</td>
      <td>AND</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Arab World</td>
      <td>ARB</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>11.914508</td>
      <td>12.160385</td>
      <td>12.918274</td>
      <td>13.025172</td>
      <td>13.516863</td>
      <td>12.477178</td>
      <td>...</td>
      <td>11.148914</td>
      <td>10.856290</td>
      <td>11.113108</td>
      <td>10.659740</td>
      <td>10.216098</td>
      <td>11.325613</td>
      <td>10.889659</td>
      <td>9.953571</td>
      <td>9.581397</td>
      <td>9.461238</td>
    </tr>
    <tr>
      <th>8</th>
      <td>United Arab Emirates</td>
      <td>ARE</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>1.625000</td>
      <td>1.713000</td>
      <td>1.905000</td>
      <td>1.836000</td>
      <td>1.800000</td>
      <td>1.834000</td>
      <td>...</td>
      <td>1.793000</td>
      <td>1.640000</td>
      <td>2.462000</td>
      <td>2.236000</td>
      <td>2.331000</td>
      <td>4.294000</td>
      <td>3.105000</td>
      <td>2.872000</td>
      <td>2.151000</td>
      <td>2.133000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Argentina</td>
      <td>ARG</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>5.440000</td>
      <td>6.360000</td>
      <td>10.100000</td>
      <td>11.760000</td>
      <td>18.800000</td>
      <td>17.110000</td>
      <td>...</td>
      <td>7.579000</td>
      <td>8.085000</td>
      <td>8.347000</td>
      <td>9.220000</td>
      <td>9.843000</td>
      <td>11.461000</td>
      <td>8.736000</td>
      <td>6.805000</td>
      <td>6.139000</td>
      <td>7.876000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Armenia</td>
      <td>ARM</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>1.783000</td>
      <td>1.800000</td>
      <td>5.300000</td>
      <td>6.600000</td>
      <td>6.700000</td>
      <td>9.300000</td>
      <td>...</td>
      <td>18.261000</td>
      <td>17.617000</td>
      <td>17.704000</td>
      <td>18.966000</td>
      <td>18.304000</td>
      <td>18.175000</td>
      <td>15.469000</td>
      <td>13.379000</td>
      <td>13.245000</td>
      <td>13.329000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>American Samoa</td>
      <td>ASM</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Antigua and Barbuda</td>
      <td>ATG</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Australia</td>
      <td>AUS</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>9.586000</td>
      <td>10.733000</td>
      <td>10.879000</td>
      <td>9.724000</td>
      <td>8.473000</td>
      <td>8.509000</td>
      <td>...</td>
      <td>6.055000</td>
      <td>5.711000</td>
      <td>5.592000</td>
      <td>5.300000</td>
      <td>5.159000</td>
      <td>6.456000</td>
      <td>5.116000</td>
      <td>3.728000</td>
      <td>3.668000</td>
      <td>4.072000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Austria</td>
      <td>AUT</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>3.420000</td>
      <td>3.590000</td>
      <td>4.250000</td>
      <td>3.535000</td>
      <td>4.345000</td>
      <td>5.282000</td>
      <td>...</td>
      <td>5.802000</td>
      <td>6.064000</td>
      <td>5.561000</td>
      <td>4.933000</td>
      <td>4.560000</td>
      <td>5.201000</td>
      <td>6.459000</td>
      <td>4.992000</td>
      <td>5.264000</td>
      <td>5.439000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Azerbaijan</td>
      <td>AZE</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>0.900000</td>
      <td>1.800000</td>
      <td>4.500000</td>
      <td>6.300000</td>
      <td>7.200000</td>
      <td>8.100000</td>
      <td>...</td>
      <td>4.960000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>4.900000</td>
      <td>5.000000</td>
      <td>7.240000</td>
      <td>6.040000</td>
      <td>5.650000</td>
      <td>5.636000</td>
      <td>5.594000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Burundi</td>
      <td>BDI</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>3.158000</td>
      <td>3.133000</td>
      <td>3.080000</td>
      <td>3.095000</td>
      <td>3.077000</td>
      <td>3.076000</td>
      <td>...</td>
      <td>1.442000</td>
      <td>1.347000</td>
      <td>1.247000</td>
      <td>1.146000</td>
      <td>1.043000</td>
      <td>1.030000</td>
      <td>1.118000</td>
      <td>0.915000</td>
      <td>0.921000</td>
      <td>0.902000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Belgium</td>
      <td>BEL</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>6.984000</td>
      <td>6.702000</td>
      <td>8.078000</td>
      <td>9.645000</td>
      <td>9.337000</td>
      <td>9.483000</td>
      <td>...</td>
      <td>8.482000</td>
      <td>7.830000</td>
      <td>7.090000</td>
      <td>5.941000</td>
      <td>5.364000</td>
      <td>5.545000</td>
      <td>6.248000</td>
      <td>5.570000</td>
      <td>5.528000</td>
      <td>5.488000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Benin</td>
      <td>BEN</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>1.640000</td>
      <td>1.510000</td>
      <td>1.495000</td>
      <td>1.455000</td>
      <td>1.305000</td>
      <td>1.246000</td>
      <td>...</td>
      <td>1.820000</td>
      <td>1.784000</td>
      <td>1.659000</td>
      <td>1.410000</td>
      <td>1.214000</td>
      <td>1.502000</td>
      <td>1.779000</td>
      <td>1.685000</td>
      <td>1.657000</td>
      <td>1.722000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Burkina Faso</td>
      <td>BFA</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>2.477000</td>
      <td>2.511000</td>
      <td>2.438000</td>
      <td>2.580000</td>
      <td>2.533000</td>
      <td>2.451000</td>
      <td>...</td>
      <td>4.461000</td>
      <td>4.587000</td>
      <td>4.656000</td>
      <td>4.695000</td>
      <td>4.710000</td>
      <td>5.040000</td>
      <td>5.200000</td>
      <td>5.389000</td>
      <td>5.348000</td>
      <td>5.166000</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 38 columns</p>
</div>




```python
df_data.shape 
```




    (266, 38)




```python
df_data.columns
```




    Index(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code',
           '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
           '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
           '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
           '2018', '2019', '2020', '2021', '2022', '2023', '2024'],
          dtype='object')




```python
df_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 266 entries, 0 to 265
    Data columns (total 38 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Country Name    266 non-null    object 
     1   Country Code    266 non-null    object 
     2   Indicator Name  266 non-null    object 
     3   Indicator Code  266 non-null    object 
     4   1991            235 non-null    float64
     5   1992            235 non-null    float64
     6   1993            235 non-null    float64
     7   1994            235 non-null    float64
     8   1995            235 non-null    float64
     9   1996            235 non-null    float64
     10  1997            235 non-null    float64
     11  1998            235 non-null    float64
     12  1999            235 non-null    float64
     13  2000            235 non-null    float64
     14  2001            235 non-null    float64
     15  2002            235 non-null    float64
     16  2003            235 non-null    float64
     17  2004            235 non-null    float64
     18  2005            235 non-null    float64
     19  2006            235 non-null    float64
     20  2007            235 non-null    float64
     21  2008            235 non-null    float64
     22  2009            235 non-null    float64
     23  2010            235 non-null    float64
     24  2011            235 non-null    float64
     25  2012            235 non-null    float64
     26  2013            235 non-null    float64
     27  2014            235 non-null    float64
     28  2015            235 non-null    float64
     29  2016            235 non-null    float64
     30  2017            235 non-null    float64
     31  2018            235 non-null    float64
     32  2019            235 non-null    float64
     33  2020            235 non-null    float64
     34  2021            235 non-null    float64
     35  2022            234 non-null    float64
     36  2023            232 non-null    float64
     37  2024            230 non-null    float64
    dtypes: float64(34), object(4)
    memory usage: 79.1+ KB
    


```python
df_data.dtypes
```




    Country Name       object
    Country Code       object
    Indicator Name     object
    Indicator Code     object
    1991              float64
    1992              float64
    1993              float64
    1994              float64
    1995              float64
    1996              float64
    1997              float64
    1998              float64
    1999              float64
    2000              float64
    2001              float64
    2002              float64
    2003              float64
    2004              float64
    2005              float64
    2006              float64
    2007              float64
    2008              float64
    2009              float64
    2010              float64
    2011              float64
    2012              float64
    2013              float64
    2014              float64
    2015              float64
    2016              float64
    2017              float64
    2018              float64
    2019              float64
    2020              float64
    2021              float64
    2022              float64
    2023              float64
    2024              float64
    dtype: object




```python
df_data.tail()
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
      <th>Country Name</th>
      <th>Country Code</th>
      <th>Indicator Name</th>
      <th>Indicator Code</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>...</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
      <th>2024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>261</th>
      <td>Kosovo</td>
      <td>XKX</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>262</th>
      <td>Yemen, Rep.</td>
      <td>YEM</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>8.409</td>
      <td>8.342</td>
      <td>8.344</td>
      <td>8.340</td>
      <td>8.988</td>
      <td>9.585</td>
      <td>...</td>
      <td>17.900</td>
      <td>18.416</td>
      <td>18.603</td>
      <td>17.584</td>
      <td>17.202</td>
      <td>17.953</td>
      <td>18.287</td>
      <td>17.363</td>
      <td>17.091</td>
      <td>17.086</td>
    </tr>
    <tr>
      <th>263</th>
      <td>South Africa</td>
      <td>ZAF</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>23.002</td>
      <td>23.262</td>
      <td>23.179</td>
      <td>22.942</td>
      <td>22.647</td>
      <td>22.480</td>
      <td>...</td>
      <td>25.149</td>
      <td>26.536</td>
      <td>27.035</td>
      <td>26.906</td>
      <td>28.468</td>
      <td>29.217</td>
      <td>34.007</td>
      <td>33.268</td>
      <td>32.098</td>
      <td>33.168</td>
    </tr>
    <tr>
      <th>264</th>
      <td>Zambia</td>
      <td>ZMB</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>18.900</td>
      <td>19.544</td>
      <td>19.700</td>
      <td>18.648</td>
      <td>16.828</td>
      <td>15.300</td>
      <td>...</td>
      <td>5.942</td>
      <td>5.239</td>
      <td>4.529</td>
      <td>5.033</td>
      <td>5.542</td>
      <td>6.033</td>
      <td>5.199</td>
      <td>5.995</td>
      <td>5.905</td>
      <td>5.961</td>
    </tr>
    <tr>
      <th>265</th>
      <td>Zimbabwe</td>
      <td>ZWE</td>
      <td>Unemployment, total (% of total labor force) (...</td>
      <td>SL.UEM.TOTL.ZS</td>
      <td>4.813</td>
      <td>4.938</td>
      <td>4.990</td>
      <td>4.960</td>
      <td>5.571</td>
      <td>6.163</td>
      <td>...</td>
      <td>5.377</td>
      <td>5.886</td>
      <td>6.344</td>
      <td>6.793</td>
      <td>7.373</td>
      <td>8.621</td>
      <td>9.540</td>
      <td>10.087</td>
      <td>8.759</td>
      <td>8.554</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
df_data.describe()
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
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
      <th>2000</th>
      <th>...</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
      <th>2024</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>...</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>235.000000</td>
      <td>234.000000</td>
      <td>232.000000</td>
      <td>230.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.309450</td>
      <td>7.657994</td>
      <td>8.077971</td>
      <td>8.228115</td>
      <td>8.271647</td>
      <td>8.382771</td>
      <td>8.268826</td>
      <td>8.305579</td>
      <td>8.456762</td>
      <td>8.333982</td>
      <td>...</td>
      <td>7.802640</td>
      <td>7.667339</td>
      <td>7.452974</td>
      <td>7.206506</td>
      <td>7.064106</td>
      <td>8.121189</td>
      <td>7.809688</td>
      <td>6.949933</td>
      <td>6.606764</td>
      <td>6.521012</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.565850</td>
      <td>5.866918</td>
      <td>5.923331</td>
      <td>5.833094</td>
      <td>5.893925</td>
      <td>5.899470</td>
      <td>5.734063</td>
      <td>5.659025</td>
      <td>5.668398</td>
      <td>5.699568</td>
      <td>...</td>
      <td>5.474281</td>
      <td>5.369211</td>
      <td>5.253675</td>
      <td>5.205504</td>
      <td>5.132942</td>
      <td>5.498414</td>
      <td>5.491331</td>
      <td>5.285512</td>
      <td>5.040455</td>
      <td>5.007064</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.600000</td>
      <td>0.596000</td>
      <td>0.610000</td>
      <td>0.617000</td>
      <td>0.632000</td>
      <td>0.616000</td>
      <td>0.631000</td>
      <td>0.645000</td>
      <td>0.652000</td>
      <td>0.634000</td>
      <td>...</td>
      <td>0.170000</td>
      <td>0.150000</td>
      <td>0.140000</td>
      <td>0.110000</td>
      <td>0.100000</td>
      <td>0.140000</td>
      <td>0.140000</td>
      <td>0.130000</td>
      <td>0.130000</td>
      <td>0.126000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.091500</td>
      <td>3.369500</td>
      <td>3.877935</td>
      <td>4.072500</td>
      <td>4.130000</td>
      <td>4.231500</td>
      <td>4.176439</td>
      <td>4.296083</td>
      <td>4.443411</td>
      <td>4.295584</td>
      <td>...</td>
      <td>4.227734</td>
      <td>4.199787</td>
      <td>4.016000</td>
      <td>3.821267</td>
      <td>3.707500</td>
      <td>4.455500</td>
      <td>4.452500</td>
      <td>3.728500</td>
      <td>3.563250</td>
      <td>3.434000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.079498</td>
      <td>6.316000</td>
      <td>6.346000</td>
      <td>6.889000</td>
      <td>7.157000</td>
      <td>7.222000</td>
      <td>7.079000</td>
      <td>7.187000</td>
      <td>7.112627</td>
      <td>6.776076</td>
      <td>...</td>
      <td>6.313000</td>
      <td>6.029099</td>
      <td>5.833000</td>
      <td>5.511000</td>
      <td>5.552000</td>
      <td>6.690897</td>
      <td>6.193192</td>
      <td>5.469230</td>
      <td>5.183500</td>
      <td>5.077229</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.574624</td>
      <td>9.951500</td>
      <td>10.687545</td>
      <td>10.689339</td>
      <td>10.490233</td>
      <td>11.011518</td>
      <td>10.850000</td>
      <td>11.168500</td>
      <td>11.654000</td>
      <td>11.118000</td>
      <td>...</td>
      <td>9.838000</td>
      <td>9.661000</td>
      <td>9.239780</td>
      <td>8.778789</td>
      <td>8.833500</td>
      <td>9.894745</td>
      <td>9.469563</td>
      <td>8.411122</td>
      <td>7.979966</td>
      <td>7.810413</td>
    </tr>
    <tr>
      <th>max</th>
      <td>30.228000</td>
      <td>30.283000</td>
      <td>30.348000</td>
      <td>30.334000</td>
      <td>35.600000</td>
      <td>38.800000</td>
      <td>36.000000</td>
      <td>34.500000</td>
      <td>32.400000</td>
      <td>32.200000</td>
      <td>...</td>
      <td>27.695000</td>
      <td>26.536000</td>
      <td>27.035000</td>
      <td>26.906000</td>
      <td>28.468000</td>
      <td>32.944000</td>
      <td>34.153000</td>
      <td>35.359000</td>
      <td>35.086000</td>
      <td>34.400000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 34 columns</p>
</div>



## Step 2: Data Cleaning

The dataset is cleaned to address missing values and outliers, ensuring data quality for analysis and modeling. Missing value rates of unemployment are calculated to assess data completeness.


```python
# Define columns that are years (e.g., 1990 to 2024)
year_cols = [col for col in df_data.columns if col.isdigit()]

# Countries where all year columns are missing (NaN)
all_missing = df_data[df_data[year_cols].isnull().all(axis=1)]

# Show total count
num_all_missing = all_missing.shape[0]
print(f"Total number of countries missing ALL unemployment year data rate: {num_all_missing}")

# Optionally, show their names
all_missing[['Country Name']]



```

    Total number of countries missing ALL unemployment year data rate: 31
    




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
      <th>Country Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Andorra</td>
    </tr>
    <tr>
      <th>11</th>
      <td>American Samoa</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Antigua and Barbuda</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Bermuda</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Curacao</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Cayman Islands</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Dominica</td>
    </tr>
    <tr>
      <th>78</th>
      <td>Faroe Islands</td>
    </tr>
    <tr>
      <th>79</th>
      <td>Micronesia, Fed. Sts.</td>
    </tr>
    <tr>
      <th>84</th>
      <td>Gibraltar</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Grenada</td>
    </tr>
    <tr>
      <th>91</th>
      <td>Greenland</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Isle of Man</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Not classified</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Kiribati</td>
    </tr>
    <tr>
      <th>125</th>
      <td>St. Kitts and Nevis</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Liechtenstein</td>
    </tr>
    <tr>
      <th>147</th>
      <td>St. Martin (French part)</td>
    </tr>
    <tr>
      <th>149</th>
      <td>Monaco</td>
    </tr>
    <tr>
      <th>155</th>
      <td>Marshall Islands</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Northern Mariana Islands</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Nauru</td>
    </tr>
    <tr>
      <th>188</th>
      <td>Palau</td>
    </tr>
    <tr>
      <th>212</th>
      <td>San Marino</td>
    </tr>
    <tr>
      <th>225</th>
      <td>Sint Maarten (Dutch part)</td>
    </tr>
    <tr>
      <th>226</th>
      <td>Seychelles</td>
    </tr>
    <tr>
      <th>228</th>
      <td>Turks and Caicos Islands</td>
    </tr>
    <tr>
      <th>245</th>
      <td>Tuvalu</td>
    </tr>
    <tr>
      <th>255</th>
      <td>British Virgin Islands</td>
    </tr>
    <tr>
      <th>261</th>
      <td>Kosovo</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ensure 'Missing_Years' is computed
df_data['Missing_Years'] = df_data[year_cols].isnull().sum(axis=1)

# Select countries with some (but not all) missing years
partial_missing = df_data[
    (df_data['Missing_Years'] > 0) & (df_data['Missing_Years'] < len(year_cols))
][['Country Name', 'Missing_Years']]

# Total number of such countries
num_partial_missing = partial_missing.shape[0]
print(f"Total number of countries with PARTIALLY missing unemployment data rate: {num_partial_missing}")

# Optionally, display the top 20
partial_missing = partial_missing.sort_values(by='Missing_Years', ascending=False)
partial_missing.head(20)

```

    Total number of countries with PARTIALLY missing unemployment data rate: 5
    




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
      <th>Country Name</th>
      <th>Missing_Years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>248</th>
      <td>Ukraine</td>
      <td>3</td>
    </tr>
    <tr>
      <th>196</th>
      <td>West Bank and Gaza</td>
      <td>2</td>
    </tr>
    <tr>
      <th>206</th>
      <td>Sudan</td>
      <td>2</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Lebanon</td>
      <td>1</td>
    </tr>
    <tr>
      <th>216</th>
      <td>South Sudan</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Select countries with no missing unemployment rates
countries_full_data = df_data[df_data['Missing_Years'] == 0][['Country Name']]

# Print the exact number of such countries
print(f"Countries with COMPLETE unemployment data rate: {len(countries_full_data)}")

# Optionally, display the first 20 country names
countries_full_data.head(20)


```

    Countries with COMPLETE unemployment data rate: 230
    




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
      <th>Country Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Africa Eastern and Southern</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Africa Western and Central</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Albania</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Arab World</td>
    </tr>
    <tr>
      <th>8</th>
      <td>United Arab Emirates</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Argentina</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Armenia</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Australia</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Austria</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Azerbaijan</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Burundi</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Belgium</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Benin</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Burkina Faso</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Bangladesh</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Bulgaria</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Bahrain</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Bahamas, The</td>
    </tr>
  </tbody>
</table>
</div>



##### Handle Missing Data

##### 2.1 Drop Countries with all missing Data and Handling Outliers


```python
# Drop countries missing ALL unemployment year data rate

year_cols = [col for col in df_data.columns if col.isdigit()]
df_cleaned = df_data.dropna(subset=year_cols, how='all').copy()


```


```python
# Verify the number of remaining rows

print(f"Shape after removing countries with all missing data: {df_cleaned.shape}")
```

    Shape after removing countries with all missing data: (235, 39)
    

##### 2.2 Handle Partially Missing Data


```python
# Impute missing values using linear interpolation for each country

df_cleaned[year_cols] = df_cleaned[year_cols].interpolate(method='linear', axis=1, limit_direction='both')

# Check if any missing values remain in year columns
print(f"Remaining missing values in year columns: {df_cleaned[year_cols].isnull().sum().sum()}")


```

    Remaining missing values in year columns: 0
    


```python
# Verify no missing values remain

print(f"Final check for missing values: {df_cleaned[year_cols].isnull().sum().sum()}")
```

    Final check for missing values: 0
    

##### 2.3 Drop Unnecessary/Redundant Columns


```python
# Drop non-essential columns
df_cleaned = df_cleaned.drop(columns=['Indicator Name', 'Indicator Code', 'Missing_Years'])

# Verify remaining columns
print(df_cleaned.columns)
```

    Index(['Country Name', 'Country Code', '1991', '1992', '1993', '1994', '1995',
           '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
           '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
           '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022',
           '2023', '2024'],
          dtype='object')
    

##### 2.4 Detect Outliers


```python
# Drop rows with any NaN in year columns before outlier detection
df_no_nan = df_cleaned.dropna(subset=year_cols)

print(f"Shape before outlier detection: {df_no_nan.shape}")

```

    Shape before outlier detection: (235, 36)
    


```python
# Outlier Detection and Capping for Unemployment Rates
# 1. Standardize and detect outlier countries using Isolation Forest
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_no_nan[year_cols])

iso_forest = IsolationForest(contamination=0.05, random_state=42)
outlier_labels = iso_forest.fit_predict(scaled_data)
outliers = df_no_nan[outlier_labels == -1].copy()
print(f"Number of outlier countries: {len(outliers)}")
print("Outlier countries:\n", outliers[['Country Name']])

# 2. Compute capping thresholds for each year (5th/95th percentiles)
cap_lower = df_no_nan[year_cols].quantile(0.05)
cap_upper = df_no_nan[year_cols].quantile(0.95)

# 3. Build a summary table: only rows where capping changed the value
records = []
for idx, row in outliers.iterrows():
    country = row['Country Name']
    for year in year_cols:
        orig_val = row[year]
        if pd.isnull(orig_val):
            continue
        capped_val = min(max(orig_val, cap_lower[year]), cap_upper[year])
        cap_applied = ''
        if orig_val < cap_lower[year]:
            cap_applied = 'Lower'
        elif orig_val > cap_upper[year]:
            cap_applied = 'Upper'
        if cap_applied:  # Only record rows where capping was applied
            records.append({
                'Country': country,
                'Year': year,
                'Original Value': orig_val,
                'Capped Value': capped_val,
                'Cap Applied': cap_applied
            })

summary_df = pd.DataFrame(records)

# 4. Count unique countries affected and print list
num_countries_affected = summary_df['Country'].nunique()
countries_list = summary_df['Country'].unique()

print(f"\nNumber of unique countries affected (with capped outlier values): {num_countries_affected}")
print(f"List of affected countries:\n{countries_list}")

# 5. Display ALL rows of the result (not recommended for huge outputs in notebook)
pd.set_option('display.max_rows', None)
print(summary_df)
pd.reset_option('display.max_rows')

# 6. Save as CSV for easier viewing:
summary_df.to_csv("outlier_capping_summary.csv", index=False)
print("Saved to outlier_capping_summary.csv")

```

    Number of outlier countries: 12
    Outlier countries:
                    Country Name
    24   Bosnia and Herzegovina
    33                 Botswana
    56                 Djibouti
    60                  Algeria
    70                    Spain
    89                   Greece
    157         North Macedonia
    162              Montenegro
    171                 Namibia
    196      West Bank and Gaza
    224                Eswatini
    263            South Africa
    
    Number of unique countries affected (with capped outlier values): 12
    List of affected countries:
    ['Bosnia and Herzegovina' 'Botswana' 'Djibouti' 'Algeria' 'Spain' 'Greece'
     'North Macedonia' 'Montenegro' 'Namibia' 'West Bank and Gaza' 'Eswatini'
     'South Africa']
                        Country  Year  Original Value  Capped Value Cap Applied
    0    Bosnia and Herzegovina  1994          20.383       20.0514       Upper
    1    Bosnia and Herzegovina  1995          21.078       19.8211       Upper
    2    Bosnia and Herzegovina  1996          21.725       20.4473       Upper
    3    Bosnia and Herzegovina  1997          22.378       19.9550       Upper
    4    Bosnia and Herzegovina  1998          23.295       19.9391       Upper
    5    Bosnia and Herzegovina  1999          24.284       19.5518       Upper
    6    Bosnia and Herzegovina  2000          25.187       19.0764       Upper
    7    Bosnia and Herzegovina  2001          26.289       19.4227       Upper
    8    Bosnia and Herzegovina  2002          27.312       20.2647       Upper
    9    Bosnia and Herzegovina  2003          28.447       19.9194       Upper
    10   Bosnia and Herzegovina  2004          29.276       19.1513       Upper
    11   Bosnia and Herzegovina  2005          30.227       19.1326       Upper
    12   Bosnia and Herzegovina  2006          31.110       18.8459       Upper
    13   Bosnia and Herzegovina  2007          28.984       18.2673       Upper
    14   Bosnia and Herzegovina  2008          23.405       18.1460       Upper
    15   Bosnia and Herzegovina  2009          24.068       19.2614       Upper
    16   Bosnia and Herzegovina  2010          27.312       19.5608       Upper
    17   Bosnia and Herzegovina  2011          27.582       19.4766       Upper
    18   Bosnia and Herzegovina  2012          28.010       19.5543       Upper
    19   Bosnia and Herzegovina  2013          27.490       19.6477       Upper
    20   Bosnia and Herzegovina  2014          27.517       19.2667       Upper
    21   Bosnia and Herzegovina  2015          27.695       19.7666       Upper
    22   Bosnia and Herzegovina  2016          25.408       19.5559       Upper
    23   Bosnia and Herzegovina  2017          20.527       19.0548       Upper
    24                 Botswana  1994          21.200       20.0514       Upper
    25                 Botswana  1995          21.394       19.8211       Upper
    26                 Botswana  1996          21.593       20.4473       Upper
    27                 Botswana  1997          21.048       19.9550       Upper
    28                 Botswana  1998          20.860       19.9391       Upper
    29                 Botswana  2002          21.268       20.2647       Upper
    30                 Botswana  2003          23.800       19.9194       Upper
    31                 Botswana  2004          21.758       19.1513       Upper
    32                 Botswana  2005          19.847       19.1326       Upper
    33                 Botswana  2017          19.664       19.0548       Upper
    34                 Botswana  2018          19.681       18.9714       Upper
    35                 Botswana  2019          20.094       17.6810       Upper
    36                 Botswana  2020          21.017       19.2730       Upper
    37                 Botswana  2021          23.106       19.6473       Upper
    38                 Botswana  2022          23.615       17.6141       Upper
    39                 Botswana  2023          23.381       17.3544       Upper
    40                 Botswana  2024          23.138       17.3590       Upper
    41                 Djibouti  1991          25.669       18.9720       Upper
    42                 Djibouti  1992          25.985       19.5443       Upper
    43                 Djibouti  1993          26.728       19.7000       Upper
    44                 Djibouti  1994          27.103       20.0514       Upper
    45                 Djibouti  1995          27.279       19.8211       Upper
    46                 Djibouti  1996          27.153       20.4473       Upper
    47                 Djibouti  1997          27.156       19.9550       Upper
    48                 Djibouti  1998          27.013       19.9391       Upper
    49                 Djibouti  1999          26.740       19.5518       Upper
    50                 Djibouti  2000          26.664       19.0764       Upper
    51                 Djibouti  2001          26.599       19.4227       Upper
    52                 Djibouti  2002          26.590       20.2647       Upper
    53                 Djibouti  2003          26.488       19.9194       Upper
    54                 Djibouti  2004          26.446       19.1513       Upper
    55                 Djibouti  2005          26.452       19.1326       Upper
    56                 Djibouti  2006          26.387       18.8459       Upper
    57                 Djibouti  2007          26.300       18.2673       Upper
    58                 Djibouti  2008          26.194       18.1460       Upper
    59                 Djibouti  2009          26.330       19.2614       Upper
    60                 Djibouti  2010          26.371       19.5608       Upper
    61                 Djibouti  2011          26.314       19.4766       Upper
    62                 Djibouti  2012          26.187       19.5543       Upper
    63                 Djibouti  2013          26.157       19.6477       Upper
    64                 Djibouti  2014          26.173       19.2667       Upper
    65                 Djibouti  2015          26.070       19.7666       Upper
    66                 Djibouti  2016          25.992       19.5559       Upper
    67                 Djibouti  2017          26.064       19.0548       Upper
    68                 Djibouti  2018          26.184       18.9714       Upper
    69                 Djibouti  2019          26.256       17.6810       Upper
    70                 Djibouti  2020          27.713       19.2730       Upper
    71                 Djibouti  2021          27.668       19.6473       Upper
    72                 Djibouti  2022          26.307       17.6141       Upper
    73                 Djibouti  2023          26.154       17.3544       Upper
    74                 Djibouti  2024          25.875       17.3590       Upper
    75                  Algeria  1991          20.600       18.9720       Upper
    76                  Algeria  1992          24.380       19.5443       Upper
    77                  Algeria  1993          26.230       19.7000       Upper
    78                  Algeria  1994          27.740       20.0514       Upper
    79                  Algeria  1995          31.840       19.8211       Upper
    80                  Algeria  1996          28.308       20.4473       Upper
    81                  Algeria  1997          25.430       19.9550       Upper
    82                  Algeria  1998          26.853       19.9391       Upper
    83                  Algeria  1999          28.548       19.5518       Upper
    84                  Algeria  2000          29.770       19.0764       Upper
    85                  Algeria  2001          27.300       19.4227       Upper
    86                  Algeria  2002          25.900       20.2647       Upper
    87                  Algeria  2003          23.720       19.9194       Upper
    88                    Spain  1993          22.156       19.7000       Upper
    89                    Spain  1994          24.209       20.0514       Upper
    90                    Spain  1995          22.675       19.8211       Upper
    91                    Spain  1996          22.142       20.4473       Upper
    92                    Spain  1997          20.698       19.9550       Upper
    93                    Spain  2010          19.860       19.5608       Upper
    94                    Spain  2011          21.390       19.4766       Upper
    95                    Spain  2012          24.789       19.5543       Upper
    96                    Spain  2013          26.094       19.6477       Upper
    97                    Spain  2014          24.441       19.2667       Upper
    98                    Spain  2015          22.057       19.7666       Upper
    99                    Spain  2016          19.635       19.5559       Upper
    100                  Greece  2012          24.731       19.5543       Upper
    101                  Greece  2013          27.686       19.6477       Upper
    102                  Greece  2014          26.708       19.2667       Upper
    103                  Greece  2015          24.981       19.7666       Upper
    104                  Greece  2016          23.514       19.5559       Upper
    105                  Greece  2017          21.413       19.0548       Upper
    106                  Greece  2018          19.179       18.9714       Upper
    107         North Macedonia  1991          24.500       18.9720       Upper
    108         North Macedonia  1992          26.300       19.5443       Upper
    109         North Macedonia  1993          27.700       19.7000       Upper
    110         North Macedonia  1994          30.000       20.0514       Upper
    111         North Macedonia  1995          35.600       19.8211       Upper
    112         North Macedonia  1996          38.800       20.4473       Upper
    113         North Macedonia  1997          36.000       19.9550       Upper
    114         North Macedonia  1998          34.500       19.9391       Upper
    115         North Macedonia  1999          32.400       19.5518       Upper
    116         North Macedonia  2000          32.200       19.0764       Upper
    117         North Macedonia  2001          30.520       19.4227       Upper
    118         North Macedonia  2002          31.940       20.2647       Upper
    119         North Macedonia  2003          36.690       19.9194       Upper
    120         North Macedonia  2004          37.161       19.1513       Upper
    121         North Macedonia  2005          37.320       19.1326       Upper
    122         North Macedonia  2006          36.392       18.8459       Upper
    123         North Macedonia  2007          35.231       18.2673       Upper
    124         North Macedonia  2008          33.930       18.1460       Upper
    125         North Macedonia  2009          32.351       19.2614       Upper
    126         North Macedonia  2010          33.135       19.5608       Upper
    127         North Macedonia  2011          31.502       19.4766       Upper
    128         North Macedonia  2012          31.096       19.5543       Upper
    129         North Macedonia  2013          29.017       19.6477       Upper
    130         North Macedonia  2014          28.215       19.2667       Upper
    131         North Macedonia  2015          26.395       19.7666       Upper
    132         North Macedonia  2016          24.312       19.5559       Upper
    133         North Macedonia  2017          22.857       19.0548       Upper
    134         North Macedonia  2018          21.208       18.9714       Upper
    135              Montenegro  1991          30.228       18.9720       Upper
    136              Montenegro  1992          30.283       19.5443       Upper
    137              Montenegro  1993          30.348       19.7000       Upper
    138              Montenegro  1994          30.334       20.0514       Upper
    139              Montenegro  1995          30.294       19.8211       Upper
    140              Montenegro  1996          30.235       20.4473       Upper
    141              Montenegro  1997          30.298       19.9550       Upper
    142              Montenegro  1998          30.269       19.9391       Upper
    143              Montenegro  1999          30.676       19.5518       Upper
    144              Montenegro  2000          30.609       19.0764       Upper
    145              Montenegro  2001          30.741       19.4227       Upper
    146              Montenegro  2002          30.423       20.2647       Upper
    147              Montenegro  2003          30.458       19.9194       Upper
    148              Montenegro  2004          30.370       19.1513       Upper
    149              Montenegro  2005          30.310       19.1326       Upper
    150              Montenegro  2006          24.790       18.8459       Upper
    151              Montenegro  2007          19.400       18.2673       Upper
    152              Montenegro  2010          19.649       19.5608       Upper
    153              Montenegro  2011          19.759       19.4766       Upper
    154              Montenegro  2012          19.808       19.5543       Upper
    155                 Namibia  1991          19.140       18.9720       Upper
    156                 Namibia  1995          21.312       19.8211       Upper
    157                 Namibia  1996          22.835       20.4473       Upper
    158                 Namibia  1997          24.450       19.9550       Upper
    159                 Namibia  1998          23.081       19.9391       Upper
    160                 Namibia  1999          21.669       19.5518       Upper
    161                 Namibia  2000          20.300       19.0764       Upper
    162                 Namibia  2001          20.980       19.4227       Upper
    163                 Namibia  2002          21.500       20.2647       Upper
    164                 Namibia  2003          22.052       19.9194       Upper
    165                 Namibia  2004          22.090       19.1513       Upper
    166                 Namibia  2005          22.107       19.1326       Upper
    167                 Namibia  2006          21.876       18.8459       Upper
    168                 Namibia  2007          22.119       18.2673       Upper
    169                 Namibia  2008          22.017       18.1460       Upper
    170                 Namibia  2009          22.254       19.2614       Upper
    171                 Namibia  2010          22.120       19.5608       Upper
    172                 Namibia  2015          20.808       19.7666       Upper
    173                 Namibia  2016          23.352       19.5559       Upper
    174                 Namibia  2017          21.733       19.0548       Upper
    175                 Namibia  2018          19.877       18.9714       Upper
    176                 Namibia  2019          19.921       17.6810       Upper
    177                 Namibia  2020          21.004       19.2730       Upper
    178                 Namibia  2021          20.922       19.6473       Upper
    179                 Namibia  2022          19.695       17.6141       Upper
    180                 Namibia  2023          19.365       17.3544       Upper
    181                 Namibia  2024          19.148       17.3590       Upper
    182      West Bank and Gaza  2001          21.493       19.4227       Upper
    183      West Bank and Gaza  2002          27.465       20.2647       Upper
    184      West Bank and Gaza  2003          23.004       19.9194       Upper
    185      West Bank and Gaza  2004          23.215       19.1513       Upper
    186      West Bank and Gaza  2005          20.016       19.1326       Upper
    187      West Bank and Gaza  2006          19.014       18.8459       Upper
    188      West Bank and Gaza  2007          18.282       18.2673       Upper
    189      West Bank and Gaza  2008          22.913       18.1460       Upper
    190      West Bank and Gaza  2009          20.452       19.2614       Upper
    191      West Bank and Gaza  2010          21.416       19.5608       Upper
    192      West Bank and Gaza  2013          19.894       19.6477       Upper
    193      West Bank and Gaza  2014          20.526       19.2667       Upper
    194      West Bank and Gaza  2015          23.005       19.7666       Upper
    195      West Bank and Gaza  2016          23.939       19.5559       Upper
    196      West Bank and Gaza  2017          25.677       19.0548       Upper
    197      West Bank and Gaza  2018          26.256       18.9714       Upper
    198      West Bank and Gaza  2019          25.340       17.6810       Upper
    199      West Bank and Gaza  2020          25.895       19.2730       Upper
    200      West Bank and Gaza  2021          26.390       19.6473       Upper
    201      West Bank and Gaza  2022          24.420       17.6141       Upper
    202      West Bank and Gaza  2023          24.420       17.3544       Upper
    203      West Bank and Gaza  2024          24.420       17.3590       Upper
    204                Eswatini  1991          20.684       18.9720       Upper
    205                Eswatini  1992          21.052       19.5443       Upper
    206                Eswatini  1993          21.750       19.7000       Upper
    207                Eswatini  1994          21.717       20.0514       Upper
    208                Eswatini  1995          21.650       19.8211       Upper
    209                Eswatini  1996          22.074       20.4473       Upper
    210                Eswatini  1997          22.500       19.9550       Upper
    211                Eswatini  1998          23.195       19.9391       Upper
    212                Eswatini  1999          23.837       19.5518       Upper
    213                Eswatini  2000          24.502       19.0764       Upper
    214                Eswatini  2001          25.178       19.4227       Upper
    215                Eswatini  2002          25.716       20.2647       Upper
    216                Eswatini  2003          26.216       19.9194       Upper
    217                Eswatini  2004          26.690       19.1513       Upper
    218                Eswatini  2005          27.204       19.1326       Upper
    219                Eswatini  2006          27.690       18.8459       Upper
    220                Eswatini  2007          28.240       18.2673       Upper
    221                Eswatini  2008          27.835       18.1460       Upper
    222                Eswatini  2009          27.383       19.2614       Upper
    223                Eswatini  2010          26.736       19.5608       Upper
    224                Eswatini  2011          25.987       19.4766       Upper
    225                Eswatini  2012          25.128       19.5543       Upper
    226                Eswatini  2013          24.448       19.6477       Upper
    227                Eswatini  2014          23.828       19.2667       Upper
    228                Eswatini  2015          23.285       19.7666       Upper
    229                Eswatini  2016          22.718       19.5559       Upper
    230                Eswatini  2017          24.523       19.0548       Upper
    231                Eswatini  2018          26.369       18.9714       Upper
    232                Eswatini  2019          28.138       17.6810       Upper
    233                Eswatini  2020          32.944       19.2730       Upper
    234                Eswatini  2021          34.153       19.6473       Upper
    235                Eswatini  2022          35.359       17.6141       Upper
    236                Eswatini  2023          35.086       17.3544       Upper
    237                Eswatini  2024          34.400       17.3590       Upper
    238            South Africa  1991          23.002       18.9720       Upper
    239            South Africa  1992          23.262       19.5443       Upper
    240            South Africa  1993          23.179       19.7000       Upper
    241            South Africa  1994          22.942       20.0514       Upper
    242            South Africa  1995          22.647       19.8211       Upper
    243            South Africa  1996          22.480       20.4473       Upper
    244            South Africa  1997          22.518       19.9550       Upper
    245            South Africa  1998          22.673       19.9391       Upper
    246            South Africa  1999          22.791       19.5518       Upper
    247            South Africa  2000          22.714       19.0764       Upper
    248            South Africa  2001          22.605       19.4227       Upper
    249            South Africa  2002          22.547       20.2647       Upper
    250            South Africa  2003          22.629       19.9194       Upper
    251            South Africa  2004          22.538       19.1513       Upper
    252            South Africa  2005          22.461       19.1326       Upper
    253            South Africa  2006          22.324       18.8459       Upper
    254            South Africa  2007          22.287       18.2673       Upper
    255            South Africa  2008          22.407       18.1460       Upper
    256            South Africa  2009          23.523       19.2614       Upper
    257            South Africa  2010          24.683       19.5608       Upper
    258            South Africa  2011          24.639       19.4766       Upper
    259            South Africa  2012          24.727       19.5543       Upper
    260            South Africa  2013          24.561       19.6477       Upper
    261            South Africa  2014          24.890       19.2667       Upper
    262            South Africa  2015          25.149       19.7666       Upper
    263            South Africa  2016          26.536       19.5559       Upper
    264            South Africa  2017          27.035       19.0548       Upper
    265            South Africa  2018          26.906       18.9714       Upper
    266            South Africa  2019          28.468       17.6810       Upper
    267            South Africa  2020          29.217       19.2730       Upper
    268            South Africa  2021          34.007       19.6473       Upper
    269            South Africa  2022          33.268       17.6141       Upper
    270            South Africa  2023          32.098       17.3544       Upper
    271            South Africa  2024          33.168       17.3590       Upper
    Saved to outlier_capping_summary.csv
    

#### 2.5 Handle Outliers 


```python
# Cap outliers at the 5th and 95th percentiles
for col in year_cols:
    lower_bound = df_cleaned[col].quantile(0.05)
    upper_bound = df_cleaned[col].quantile(0.95)
    df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

# Verify the effect by checking the summary statistics
print(df_cleaned[year_cols].describe())
```

                 1991        1992        1993        1994        1995        1996  \
    count  235.000000  235.000000  235.000000  235.000000  235.000000  235.000000   
    mean     7.157257    7.472724    7.886129    8.044121    8.049359    8.164448   
    std      5.086335    5.252507    5.306188    5.212923    5.108611    5.105281   
    min      1.085300    1.350000    1.430000    1.487900    1.471000    1.475800   
    25%      3.091500    3.369500    3.877935    4.072500    4.130000    4.231500   
    50%      6.079498    6.316000    6.346000    6.889000    7.157000    7.222000   
    75%      9.574624    9.951500   10.687545   10.689339   10.490233   11.011518   
    max     18.972000   19.544300   19.700000   20.051400   19.821100   20.447300   
    
                 1997        1998        1999        2000  ...        2015  \
    count  235.000000  235.000000  235.000000  235.000000  ...  235.000000   
    mean     8.067197    8.102060    8.246348    8.107054  ...    7.660978   
    std      5.002310    4.916805    4.902273    4.858131  ...    4.898864   
    min      1.475600    1.583400    1.733100    1.912600  ...    1.790600   
    25%      4.176439    4.296083    4.443411    4.295584  ...    4.227734   
    50%      7.079000    7.187000    7.112627    6.776076  ...    6.313000   
    75%     10.850000   11.168500   11.654000   11.118000  ...    9.838000   
    max     19.955000   19.939100   19.551800   19.076400  ...   19.766600   
    
                 2016        2017        2018        2019        2020        2021  \
    count  235.000000  235.000000  235.000000  235.000000  235.000000  235.000000   
    mean     7.530405    7.313931    7.082901    6.884679    7.941354    7.626742   
    std      4.823906    4.705793    4.676957    4.405054    4.722717    4.659575   
    min      1.740800    1.687700    1.557000    1.697800    2.188400    2.097500   
    25%      4.199787    4.016000    3.821267    3.707500    4.455500    4.452500   
    50%      6.029099    5.833000    5.511000    5.552000    6.690897    6.193192   
    75%      9.661000    9.239780    8.778789    8.833500    9.894745    9.469563   
    max     19.555900   19.054800   18.971400   17.681000   19.273000   19.647300   
    
                 2022        2023        2024  
    count  235.000000  235.000000  235.000000  
    mean     6.707871    6.449804    6.416455  
    std      4.247413    4.109741    4.094695  
    min      1.640600    1.653400    1.581100  
    25%      3.729000    3.584000    3.522500  
    50%      5.484459    5.200878    5.144000  
    75%      8.428823    8.172932    7.862000  
    max     17.614100   17.354400   17.359000  
    
    [8 rows x 34 columns]
    


## Step 3: Exploratory Data Analytics (EDA)

Exploratory data analysis (EDA) uncovers temporal and regional patterns in unemployment rates, using descriptive statistics and visualizations. Descriptive statistics summarize central tendencies and variability across years and regions. A line plot shows average unemployment trends, highlighting spikes in 2008–2009 and 2020 due to the financial crisis and COVID-19. A regional line plot compares trends across regions, informing model generalization. A correlation heatmap identifies strong temporal dependencies. 

### Descriptive Analytics and Diagnostics Analytics


```python
# Melt the dataframe to long format,It reshapes your data for easy plotting and analysis across countries and years.
# Each row is a single country-year observation.

df_long = pd.melt(
    df_cleaned,
    id_vars=['Country Name', 'Country Code'],
    value_vars=year_cols,
    
    var_name='Year',
    value_name='Unemployment Rate'
)

# Convert Year and Unemployment Rate to correct types
df_long['Year'] = df_long['Year'].astype(int)
df_long['Unemployment Rate'] = df_long['Unemployment Rate'].astype(float)

# Drop rows where Unemployment Rate is missing (NaN)
df_long = df_long.dropna(subset=['Unemployment Rate'])

#  Print shape and a few summaries
print(df_long.head(20))
print(f"\nShape of long dataframe: {df_long.shape}")
print(f"Number of unique countries: {df_long['Country Name'].nunique()}")
print(f"Years available: {sorted(df_long['Year'].unique())}")
print(f"Missing values in Unemployment Rate: {df_long['Unemployment Rate'].isnull().sum()}")
print("\nSummary statistics for Unemployment Rate:")
print(df_long['Unemployment Rate'].describe())

```

                       Country Name Country Code  Year  Unemployment Rate
    0   Africa Eastern and Southern          AFE  1991           8.179629
    1                   Afghanistan          AFG  1991           8.070000
    2    Africa Western and Central          AFW  1991           4.158680
    3                        Angola          AGO  1991          16.855000
    4                       Albania          ALB  1991          10.304000
    5                    Arab World          ARB  1991          11.914508
    6          United Arab Emirates          ARE  1991           1.625000
    7                     Argentina          ARG  1991           5.440000
    8                       Armenia          ARM  1991           1.783000
    9                     Australia          AUS  1991           9.586000
    10                      Austria          AUT  1991           3.420000
    11                   Azerbaijan          AZE  1991           1.085300
    12                      Burundi          BDI  1991           3.158000
    13                      Belgium          BEL  1991           6.984000
    14                        Benin          BEN  1991           1.640000
    15                 Burkina Faso          BFA  1991           2.477000
    16                   Bangladesh          BGD  1991           2.200000
    17                     Bulgaria          BGR  1991          11.100000
    18                      Bahrain          BHR  1991           1.085300
    19                 Bahamas, The          BHS  1991          12.170000
    
    Shape of long dataframe: (7990, 4)
    Number of unique countries: 235
    Years available: [1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    Missing values in Unemployment Rate: 0
    
    Summary statistics for Unemployment Rate:
    count    7990.000000
    mean        7.635282
    std         4.820482
    min         1.085300
    25%         4.073500
    50%         6.321991
    75%        10.103718
    max        20.447300
    Name: Unemployment Rate, dtype: float64
    

### 3.0 Plot 1: Line Plot, Mean Unemployment Rate

**Descriptive Analytics**

Early Increase (1991–1999)
>
- The global average unemployment rate rises from around 7.2% (1991) to a peak just above 8.2% by the late 1990s and early 2000s.

Decline and Stability (2000–2008)
>
- A gradual decline is observed, falling to just above 7.0% by 2008.

Spikes and Recoveries (2008–2021)

- Around 2008–2010, there’s a visible increase—likely due to the global financial crisis.

- Afterward, the rate stabilizes and then generally trends downward, hitting a low around 2019.

- A sharp spike occurs around 2020, consistent with the impact of the COVID-19 pandemic on global labor markets.

Recent Steep Decline (2021–2024)
>
- After the 2020 peak, the mean unemployment rate drops rapidly, falling below 6.5% by 2024.

**Diagnostic Analytics**

Early 1990s Increase

- Reflects the aftermath of the Cold War, economic restructuring in post-Soviet states and global market transitions.

Late 1990s/Early 2000s Peak

- May result from the cumulative impact of various regional crises (e.g., Asian financial crisis, dot-com bubble).

2008–2010 Spike

- Directly linked to the global financial crisis that led to mass layoffs and increased unemployment worldwide.

2020 Spike

- Corresponds to the onset of the COVID-19 pandemic, which caused historic job losses globally.

Post-2021 Rapid Decline

- Could reflect quick recoveries in major economies post-pandemic, government interventions, or possibly changes in dataset coverage or reporting methods in recent years.


```python
# Plotting the global mean unemployment rate over time
plt.figure(figsize=(12, 6))
yearly_trend = df_long.groupby('Year')['Unemployment Rate'].mean()
plt.figure(figsize=(12, 6))
sns.lineplot(x=yearly_trend.index, y=yearly_trend.values)
plt.title('Global Average Unemployment Rate Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Unemployment Rate (%)')
plt.grid(True)
plt.tight_layout()
plt.show()



# This plot shows how the global average unemployment rate changes over time,
# highlighting key trends such as spikes during major economic crises
# (the 2008–2009 financial crisis and the 2020 pandemic).
```


    <Figure size 1200x600 with 0 Axes>



    
![png](output_38_1.png)
    


### 3.1 Plot 2: Histogram of Unemployment Rates


**Descriptive Analytics**
>
- This plot displays the distribution (spread, median, quartiles, and outliers) of unemployment rates for each year (x-axis: years, y-axis: unemployment rate).
>
Key Patterns

- The median unemployment rate is relatively stable across most years but may spike in crisis years (such as 2008–2009, 2020).

- The spread (interquartile range) indicates that some years see greater variability among countries.

- Outliers (points beyond the whiskers) highlight countries with unusually high unemployment in certain years.
>
**Diagnostic Analytics**

- Most countries typically maintain unemployment rates below 10% due to economic policy and labor market interventions.

- The right skewness can be driven by exceptional events (e.g., economic crises, wars, pandemics).




```python
plt.figure(figsize=(10, 6))
sns.histplot(df_long['Unemployment Rate'], bins=30, kde=True)
plt.title('Distribution of All Unemployment Rates')
plt.xlabel('Unemployment Rate (%)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


```


    
![png](output_40_0.png)
    


### 3.2 Plot 3: Boxplot of Unemployment Rates Over Years

**Descriptive Analytics**

- This plot displays the distribution (spread, median, quartiles, and outliers) of unemployment rates for each year (x-axis: years, y-axis: unemployment rate).

Key Patterns

- The median unemployment rate is relatively stable across most years but may spike in crisis years (such as 2008–2009, 2020).

- The spread (interquartile range) indicates that some years see greater variability among countries.

- Outliers (points beyond the whiskers) highlight countries with unusually high unemployment in certain years.
>
**Diagnostic Analytics**

What explains the variation?

- Global events (e.g., the 2008 financial crisis, COVID-19 in 2020) cause synchronized increases in unemployment.

- Variation in spread may reflect regional crises, policy changes, or structural changes in certain economies.

- Outliers often point to countries facing unique, severe economic problems (hyperinflation, political turmoil).


```python
# Boxplot to visualize the spread of unemployment rates across years
# This boxplot displays the spread of unemployment rates across years from 1991 to 2024, 
# with the x-axis representing years and the y-axis representing unemployment rates (in percentage, ranging from 2.5% to 20%).
plt.figure(figsize=(18, 7))
sns.boxplot(x='Year', y='Unemployment Rate', data=df_long)
plt.title('Yearly Spread of Unemployment Rates (Boxplot)')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

```


    
![png](output_42_0.png)
    


### 3.3 Plot 4: Small Multiples Line Chart for East African Countries.

**Descriptive Analytics**

- The chart displays year-by-year unemployment rates for each East African country from 1991 to 2024.

- Countries such as Kenya, Tanzania, and Uganda exhibit relatively stable unemployment rates, generally between 2% and 8%, with minor fluctuations over the decades.

- Ethiopia shows a gradual long-term decline, suggesting consistent labor market improvements.

- Rwanda and Burundi display relatively low unemployment rates, but with occasional small peaks.

- South Sudan and Somalia have more variable patterns, with higher spikes in certain years, possibly tied to instability.

- Pandemic year 2020 shows a small to moderate rise in unemployment for most countries, though the magnitude differs by country.

**Diagnostic Analytics**

- Stable patterns in Kenya, Tanzania, and Uganda suggest a steady economic structure or a large informal labor sector that cushions official unemployment figures.

- Ethiopia’s downward trend may be linked to infrastructure expansion, agricultural reforms, and investment-driven growth in the 2000s and 2010s.

- Rwanda’s stability may be the result of post-conflict reconstruction policies and targeted employment programs.

- Higher volatility in South Sudan and Somalia likely reflects political instability, conflict, and reliance on vulnerable economic sectors.

- The 2020 COVID-19 spike aligns with global economic disruption, though the relatively smaller jumps compared to other regions may be due to the dominance of informal work and subsistence farming, which are less affected by formal sector layoffs.


```python

year_cols = [c for c in df_cleaned.columns if c.isdigit()]

# 2) Target East African countries (edit if you want more/less)
east_africa_targets = [
    "Kenya", "Tanzania", "Uganda", "Rwanda", "Burundi",
    "South Sudan", "Ethiopia", "Somalia", "Eritrea", "Djibouti"
]

# 3) Match to names present in your dataset (case-insensitive)
name_map = {n.lower(): n for n in df_cleaned["Country Name"].unique()}
present = [name_map[n.lower()] for n in east_africa_targets if n.lower() in name_map]
missing = [n for n in east_africa_targets if n.lower() not in name_map]

if missing:
    print("Not found in dataset (check spelling or availability):", missing)
if not present:
    raise ValueError("None of the requested East African countries were found in df_cleaned.")

# 4) Long format for plotting
df_long = df_cleaned.melt(
    id_vars=["Country Name", "Country Code"],
    value_vars=year_cols,
    var_name="Year",
    value_name="Unemployment Rate"
)

# 5) Filter to East Africa and make a real copy (avoid SettingWithCopyWarning)
df_small = df_long[df_long["Country Name"].isin(present)].copy()

# 6) Clean dtypes
df_small["Year"] = df_small["Year"].astype(int)
df_small["Unemployment Rate"] = pd.to_numeric(df_small["Unemployment Rate"], errors="coerce")

# 7) Plot small multiples
g = sns.FacetGrid(
    df_small,
    col="Country Name",
    col_wrap=3,        # change to 4 if you prefer a wider grid
    height=3.6,
    sharey=False
)
g.map_dataframe(sns.lineplot, x="Year", y="Unemployment Rate", marker="o")
g.set_titles("{col_name}")
g.set_axis_labels("Year", "Unemployment Rate (%)")
plt.subplots_adjust(top=0.9)
g.fig.suptitle("Small Multiples: Unemployment Trends — East Africa", fontsize=16)
plt.show()

```


    
![png](output_44_0.png)
    


### 3.4 Plot 5: Bar Chart of Top/Bottom 50 Countries by Unemployment Rate (2024)

**Descriptive Analytics**

Shows the countries with the highest and lowest unemployment rates in 2024.

Key Patterns

- Clear distinction between regions: some regions dominate the high end, others the low.

- The top group may show rates above 20–30%, while the bottom is often under 4%.

**Diagnostic Analytics**

Factors at play?

- High unemployment often correlates with conflict, macroeconomic instability, or transition economies.

- Low unemployment may be seen in wealthy or resource-rich countries, or those with large informal sectors not captured in official statistics.

- Regional policy, labor market structure, and reporting practices all contribute.


```python
df_long = df_long.copy()
df_long['Year'] = df_long['Year'].astype(str).str.strip()
df_long['Unemployment Rate'] = pd.to_numeric(df_long['Unemployment Rate'], errors='coerce')

# --- Pick target year (prefer 2024, else latest available) ---
target_year = '2024'
years = sorted(df_long['Year'].dropna().unique(), key=lambda x: int(x))
if not years:
    raise ValueError("No valid years found in df_long['Year'].")

if target_year not in set(years):
    target_year = years[-1]

# --- Filter & sanity checks ---
df_y = (df_long.loc[df_long['Year'] == target_year, ['Country Name', 'Unemployment Rate']]
        .dropna(subset=['Unemployment Rate'])
        .sort_values('Unemployment Rate', ascending=False))

if df_y.empty:
    raise ValueError(f"No rows for Year == {target_year}. Check 'Year' and 'Unemployment Rate' columns.")

# --- Top & bottom 30 ---
top_30 = df_y.head(30).copy()
bottom_30 = df_y.tail(30).copy()

top_30['Group'] = 'Top 30 (Highest)'
bottom_30['Group'] = 'Bottom 30 (Lowest)'

extreme_countries = pd.concat([top_30, bottom_30], ignore_index=True)
extreme_countries = extreme_countries.sort_values('Unemployment Rate', ascending=False)
extreme_countries['Group'] = pd.Categorical(
    extreme_countries['Group'],
    categories=['Top 30 (Highest)', 'Bottom 30 (Lowest)'],
    ordered=True
)

# --- Plot (IMPORTANT: do NOT call plt.legend() blindly) ---
plt.figure(figsize=(14, 16))
ax = sns.barplot(
    data=extreme_countries,
    x='Unemployment Rate',
    y='Country Name',
    hue='Group',
    dodge=False,
    legend=True  # let seaborn handle the legend
)

ax.set_title(f'Top and Bottom 30 Countries by Unemployment Rate ({target_year})', fontsize=18)
ax.set_xlabel('Unemployment Rate (%)', fontsize=14)
ax.set_ylabel('Country and Regions', fontsize=14)

# If seaborn didn’t create a legend (rare in some version combos), build one manually
if ax.get_legend() is None:
    groups = extreme_countries['Group'].dropna().unique().tolist()
    handles = [Patch(label=g) for g in groups]
    ax.legend(handles=handles, title='Group', loc='best')
else:
    ax.get_legend().set_title('Group')  # just set the title on the seaborn-made legend

plt.tight_layout()
plt.show()

```


    
![png](output_46_0.png)
    


### 3.5 Plot 6: Line Plot of Global Average Unemployment Rate Over Time

**Descriptive Analytics**

- Shows how the global average unemployment rate (y-axis) changes over time (x-axis: years).

Key Patterns

- Trend,Generally stable with visible peaks and dips.

- Notable spikes during well known crises (e.g., sharp rise around 2008–2009, and possibly 2020).

- Gradual recoveries post crisis periods.

**Diagnostic Analytics**

Reasons for the trends:

- Global recessions and pandemics cause synchronous increases.

- Recoveries are often slower, reflecting labor market rigidities and lagged policy effects.

- Small fluctuations in “normal” years reflect varying regional situations.


```python
# Analyzing these provides a macro-level perspective,complementing country specific analyses.
# Plot unemployment trends for selected regions
selected_regions = [
    'Africa Eastern and Southern',
    'Africa Western and Central',
    'Arab World',
    'Central Europe and the Baltics',
    'East Asia & Pacific (excluding high income)',
    'Europe & Central Asia',
    'Euro area',
    'European Union',
    'Latin America & Caribbean',
    'Middle East, North Africa, Afghanistan & Pakistan',
    'North America',
    'OECD members',
    'South Asia',
    'Sub-Saharan Africa',
    'World'
]
df_selected = df_long[df_long['Country Name'].isin(selected_regions)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=df_selected, x='Year', y='Unemployment Rate', hue='Country Name')
plt.title('Unemployment Rate Trends for Selected Regions (1991–2024)')
plt.xlabel('Year')
plt.ylabel('Unemployment Rate (%)')
plt.legend(title='Regions', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_48_0.png)
    


### 3.6 Plot 7: Correlation Matrix of Unemployment Rates (1991–2024)

**Descriptive Analytics**

- This matrix visualizes correlations between unemployment rates across different years.

Key Patterns

- Strong positive correlation along the diagonal and between adjacent years (e.g., 2017 and 2018).

- Weaker correlations between years further apart.

**Diagnostic Analytics**

why the pattern?

- Unemployment rates tend to be “sticky” countries’ labor market conditions usually persist from year to year.

- Weakening correlation over longer intervals reflects structural changes, policy shifts, and shocks affecting labor market


```python
# Compute correlation matrix for unemployment rates across years
corr_matrix = df_cleaned[year_cols].corr()


# Full heatmap with enhanced styling
plt.figure(figsize=(16, 12))
sns.heatmap(
    corr_matrix,
    cmap='RdBu_r',  # Vibrant colormap for positive/negative correlations
    center=0,  # Center at zero for clear distinction
    annot=True,  # Show correlation values
    fmt='.2f',  # Two decimal places
    annot_kws={'size': 8, 'weight': 'bold'},  # Smaller, bold annotations
    linewidths=0.5,  # Grid lines for clarity
    linecolor='white',  # White grid lines
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8}  # Colorbar label
)
plt.title('Correlation Matrix of Unemployment Rates (1991–2024)', fontsize=18, weight='bold', pad=20)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Year', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()


```


    
![png](output_50_0.png)
    



```python
# Print key correlations for interpretation
print("\nKey Correlations:")
print(f"Correlation between 2019 and 2020: {corr_matrix.loc['2019', '2020']:.2f}")
print(f"Correlation between 2008 and 2009: {corr_matrix.loc['2008', '2009']:.2f}")


# Focused heatmap for crisis years (2008–2010, 2020–2022)
crisis_years = ['2008', '2009', '2010', '2020', '2021', '2022']
crisis_corr_matrix = df_cleaned[crisis_years].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    crisis_corr_matrix,
    cmap='RdBu_r',
    center=0,
    annot=True,
    fmt='.2f',
    annot_kws={'size': 10, 'weight': 'bold'},
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8}
)
plt.title('Correlation Matrix of Unemployment Rates (Crisis Years: 2008–2010, 2020–2022)', fontsize=14, weight='bold', pad=15)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Year', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
```

    
    Key Correlations:
    Correlation between 2019 and 2020: 0.97
    Correlation between 2008 and 2009: 0.95
    


    
![png](output_51_1.png)
    


### 3.7 Plot 8: Feature Importance (Random Forest)

**Descriptive Analytics**

- The plot ranks the past 5 years of unemployment rates (2019–2023) by their importance in predicting 2024 unemployment rates.

- 2023 dominates with the highest importance (~84%), followed far behind by 2022 (~8%), 2021 (~3%), 2020 (~2%), and 2019 (~1.9%).

- The steep drop in importance after 2023 suggests the model relies heavily on the most recent year’s data to make accurate forecasts.

- Earlier lags (2020 and 2019) contribute minimally, indicating that historical unemployment data becomes less predictive as it ages.

**Diagnostic Analytics**

- The high importance of 2023 suggests that short-term memory is critical in unemployment forecasting,recent conditions are the most reliable indicators for the next year.

- This pattern aligns with the autocorrelation often observed in economic time series, where values change gradually over time unless disrupted by major events.

- The low importance of older years implies that structural changes or shocks in the labor market make distant historical data less relevant.

- For countries with volatile or irregular unemployment trends, the model may still benefit from multiple lag years, but here, stability in trends makes the last year’s rate sufficient for strong predictions.

- Policy implication, efforts to reduce unemployment may show measurable predictive effects as early as the following year, making short-term monitoring essential.


```python
from sklearn.ensemble import RandomForestRegressor

# Prepare lag features: use past N years to predict 2024
N_lags = 5
latest_year = 2024
feature_years = [str(latest_year - i) for i in range(1, N_lags+1)]
target_year = str(latest_year)

# Drop rows with missing data for chosen years
model_df = df_cleaned[['Country Name'] + feature_years + [target_year]].dropna()
X = model_df[feature_years]
y = model_df[target_year]

# Train Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y)

# Get feature importances
importances = pd.Series(rf.feature_importances_, index=feature_years).sort_values(ascending=True)

# Plot
plt.figure(figsize=(8, 6))
importances.plot(kind='barh', color='skyblue')
plt.title(f"Feature Importance for Predicting {target_year} (Random Forest)", fontsize=14)
plt.xlabel("Importance")
plt.ylabel("Feature (Lag Year)")
plt.tight_layout()
plt.show()

importances.sort_values(ascending=False)

```


    
![png](output_53_0.png)
    





    2023    0.957085
    2022    0.025430
    2019    0.006920
    2021    0.006824
    2020    0.003742
    dtype: float64



### 3.8 Plot 9: Animated Forecast Plot

**Descriptive Analytics**

The animation visualizes unemployment rates for each country over time, with the Year progressing frame-by-frame from 1991 to 2024.

Each bubble represents a country, with:

- Y-axis: unemployment rate (%).

- Bubble size: magnitude of unemployment rate (larger = higher unemployment).

- Color: unique country identifier.

- Certain years, such as 2008–2009 and 2020, show multiple countries experiencing upward shifts in bubble position and size,indicating simultaneous unemployment spikes across the globe.

- Many countries cluster at lower unemployment levels (below 6%), while fewer countries occupy the high-rate zone (>15%).

**Diagnostic Analytics**

The synchronized spike in 2008–2009 corresponds to the global financial crisis, showing its widespread effect on labor markets worldwide.

The sharp and broad increase in 2020 aligns with the COVID-19 pandemic, which disrupted economies globally.

Post-2020, the plot reveals mixed recovery trajectories, some countries’ bubbles shrink and drop (recovery), while others remain elevated (slower rebound).

Countries consistently in the high unemployment zone may face structural labor market challenges, such as political instability, lack of diversification, or weak private sectors.

Countries with minimal movement across frames show labor market stability, possibly due to large informal sectors or strong economic resilience.


```python
import plotly.express as px

# Melt the dataframe into long format for plotly
df_long = df_cleaned.melt(id_vars=["Country Name", "Country Code"], 
                          value_vars=year_cols,
                          var_name="Year", value_name="Unemployment Rate")

# Convert Year to integer
df_long["Year"] = df_long["Year"].astype(int)

# Build animated scatter plot (showing countries' unemployment rates over time)
fig = px.scatter(df_long,
                 x="Year",
                 y="Unemployment Rate",
                 animation_frame="Year",
                 animation_group="Country Name",
                 size="Unemployment Rate",
                 color="Country Name",
                 hover_name="Country Name",
                 range_x=[1990, 2025],
                 range_y=[0, df_long["Unemployment Rate"].max() + 5],
                 title="Animated Evolution of Unemployment Rates (1991–2024)")

fig.update_layout(showlegend=False)
fig.show()

```



### 3.9 Plot 9: Ridge-Style Unemployment Rate Distributions Over Time


**Descriptive Analytics**

The plot compares global unemployment rate distributions for six key years: 1991, 2000, 2008, 2010, 2020, and 2024.

Each colored curve represents the density of unemployment rates across all countries in that year.

1991 and 2000 curves are narrower and skewed toward lower unemployment rates, indicating most countries had rates below 10%.

2008 and 2010 show a wider spread, with more countries experiencing mid-to-high unemployment rates following the global financial crisis.

2020 has a noticeable shift upward, with density peaking at higher unemployment rates,reflecting the global economic disruption caused by COVID-19.

2024’s curve shifts slightly downward compared to 2020, suggesting partial recovery but still broader variation than pre-2008.

**Descriptive Analytics**

The plot compares global unemployment rate distributions for six key years: 1991, 2000, 2008, 2010, 2020, and 2024.

Each colored curve represents the density of unemployment rates across all countries in that year.

1991 and 2000 curves are narrower and skewed toward lower unemployment rates, indicating most countries had rates below 10%.

2008 and 2010 show a wider spread, with more countries experiencing mid-to-high unemployment rates following the global financial crisis.

2020 has a noticeable shift upward, with density peaking at higher unemployment rates,reflecting the global economic disruption caused by COVID-19.

2024’s curve shifts slightly downward compared to 2020, suggesting partial recovery but still broader variation than pre-2008.






```python
# Prepare data for selected years
selected_years = ['1991', '2000', '2008', '2010', '2020', '2024']
ridge_df = df_cleaned[['Country Name'] + selected_years].melt(id_vars="Country Name",
                                                              var_name="Year",
                                                              value_name="Unemployment Rate")

# Sort years so they stack logically
year_order = sorted(selected_years, key=int)

# Create ridge-like plot using seaborn
plt.figure(figsize=(10, 7))
for i, year in enumerate(year_order):
    subset = ridge_df[ridge_df["Year"] == year]
    sns.kdeplot(subset["Unemployment Rate"], fill=True, alpha=0.6,
                label=year, linewidth=1.5)
    plt.text(x=subset["Unemployment Rate"].median(), y=i*0.02 + 0.02, s=year)

plt.title("Ridge-Style Plot: Unemployment Rate Distributions Over Time", fontsize=16)
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Density")
plt.legend(title="Year")
plt.show()

```


    
![png](output_57_0.png)
    


## Step 4: Pre-treatment for machine learning

#####  4.1 Feature Engineering,create lagged features for time-series prediction

To predict the unemployment rate for year $ t+1 $, I wll use historical unemployment rates as features. I will create a feature set for supervised learning.

I created lagged features for each country to predict unemployment rate atleast one year ahead.




```python

def create_lagged_features(df, target_year, lag_years=3):
  
    lagged_data = []
    for index, row in df.iterrows():
        country = row['Country Name']
        country_code = row['Country Code']
        for year in range(1991 + lag_years, target_year + 1):  
            features = []
            target = row[str(year)]  
            if pd.isna(target):  
                continue

            # Collect lagged features (e.g., t-1, t-2, t-3)
            for lag in range(1, lag_years + 1):
                lag_year = year - lag
                feature_value = row[str(lag_year)]
            # Skip if any feature is missing
                if pd.isna(feature_value):  
                    break
                features.append(feature_value)
            else:  
                lagged_data.append([country, country_code, year] + features + [target])
    
    columns = ['Country Name', 'Country Code', 'Year'] + [f'Lag_{i}' for i in range(1, lag_years + 1)] + ['Target']
    return pd.DataFrame(lagged_data, columns=columns)

# Create lagged features
lagged_df = create_lagged_features(df_cleaned, target_year=2024, lag_years=3)

# Verify that 'Country Code' is present
print("Columns in lagged_df:", lagged_df.columns.tolist())
print(lagged_df.head(20))
```

    Columns in lagged_df: ['Country Name', 'Country Code', 'Year', 'Lag_1', 'Lag_2', 'Lag_3', 'Target']
                       Country Name Country Code  Year     Lag_1     Lag_2  \
    0   Africa Eastern and Southern          AFE  1994  8.266327  8.270724   
    1   Africa Eastern and Southern          AFE  1995  8.138291  8.266327   
    2   Africa Eastern and Southern          AFE  1996  7.908446  8.138291   
    3   Africa Eastern and Southern          AFE  1997  7.823908  7.908446   
    4   Africa Eastern and Southern          AFE  1998  7.783654  7.823908   
    5   Africa Eastern and Southern          AFE  1999  7.812734  7.783654   
    6   Africa Eastern and Southern          AFE  2000  7.849878  7.812734   
    7   Africa Eastern and Southern          AFE  2001  7.788317  7.849878   
    8   Africa Eastern and Southern          AFE  2002  7.676955  7.788317   
    9   Africa Eastern and Southern          AFE  2003  7.632330  7.676955   
    10  Africa Eastern and Southern          AFE  2004  7.586883  7.632330   
    11  Africa Eastern and Southern          AFE  2005  7.395648  7.586883   
    12  Africa Eastern and Southern          AFE  2006  7.218793  7.395648   
    13  Africa Eastern and Southern          AFE  2007  7.158958  7.218793   
    14  Africa Eastern and Southern          AFE  2008  7.102231  7.158958   
    15  Africa Eastern and Southern          AFE  2009  7.076710  7.102231   
    16  Africa Eastern and Southern          AFE  2010  7.155881  7.076710   
    17  Africa Eastern and Southern          AFE  2011  7.403061  7.155881   
    18  Africa Eastern and Southern          AFE  2012  7.427940  7.403061   
    19  Africa Eastern and Southern          AFE  2013  7.181608  7.427940   
    
           Lag_3    Target  
    0   8.179629  8.138291  
    1   8.270724  7.908446  
    2   8.266327  7.823908  
    3   8.138291  7.783654  
    4   7.908446  7.812734  
    5   7.823908  7.849878  
    6   7.783654  7.788317  
    7   7.812734  7.676955  
    8   7.849878  7.632330  
    9   7.788317  7.586883  
    10  7.676955  7.395648  
    11  7.632330  7.218793  
    12  7.586883  7.158958  
    13  7.395648  7.102231  
    14  7.218793  7.076710  
    15  7.158958  7.155881  
    16  7.102231  7.403061  
    17  7.076710  7.427940  
    18  7.155881  7.181608  
    19  7.403061  6.986733  
    

##### 4.2 Encode Country Code

**Why We Need It**

- Machine learning models like linear regression require numerical inputs,because of that,I applied One-Hot Encoding to the Country Code column.

- "Country Code" is categorical it contains text labels (e.g., "AFG", "KEN", "USA") which models cannot directly interpret as numbers in a meaningful way.

- If you simply replace categories with numbers (e.g., AFG=1, KEN=2), the model might think there’s an order or distance between them (like "KEN" > "AFG"), which is incorrect.





```python
lagged_df = pd.get_dummies(lagged_df, columns=['Country Code'], prefix='Country')
```


```python
# Verify one-hot encoding
print("Columns after one-hot encoding:", lagged_df.columns.tolist())

print(lagged_df.head())
```

    Columns after one-hot encoding: ['Country Name', 'Year', 'Lag_1', 'Lag_2', 'Lag_3', 'Target', 'Country_AFE', 'Country_AFG', 'Country_AFW', 'Country_AGO', 'Country_ALB', 'Country_ARB', 'Country_ARE', 'Country_ARG', 'Country_ARM', 'Country_AUS', 'Country_AUT', 'Country_AZE', 'Country_BDI', 'Country_BEL', 'Country_BEN', 'Country_BFA', 'Country_BGD', 'Country_BGR', 'Country_BHR', 'Country_BHS', 'Country_BIH', 'Country_BLR', 'Country_BLZ', 'Country_BOL', 'Country_BRA', 'Country_BRB', 'Country_BRN', 'Country_BTN', 'Country_BWA', 'Country_CAF', 'Country_CAN', 'Country_CEB', 'Country_CHE', 'Country_CHI', 'Country_CHL', 'Country_CHN', 'Country_CIV', 'Country_CMR', 'Country_COD', 'Country_COG', 'Country_COL', 'Country_COM', 'Country_CPV', 'Country_CRI', 'Country_CSS', 'Country_CUB', 'Country_CYP', 'Country_CZE', 'Country_DEU', 'Country_DJI', 'Country_DNK', 'Country_DOM', 'Country_DZA', 'Country_EAP', 'Country_EAR', 'Country_EAS', 'Country_ECA', 'Country_ECS', 'Country_ECU', 'Country_EGY', 'Country_EMU', 'Country_ERI', 'Country_ESP', 'Country_EST', 'Country_ETH', 'Country_EUU', 'Country_FCS', 'Country_FIN', 'Country_FJI', 'Country_FRA', 'Country_GAB', 'Country_GBR', 'Country_GEO', 'Country_GHA', 'Country_GIN', 'Country_GMB', 'Country_GNB', 'Country_GNQ', 'Country_GRC', 'Country_GTM', 'Country_GUM', 'Country_GUY', 'Country_HIC', 'Country_HKG', 'Country_HND', 'Country_HPC', 'Country_HRV', 'Country_HTI', 'Country_HUN', 'Country_IBD', 'Country_IBT', 'Country_IDA', 'Country_IDB', 'Country_IDN', 'Country_IDX', 'Country_IND', 'Country_IRL', 'Country_IRN', 'Country_IRQ', 'Country_ISL', 'Country_ISR', 'Country_ITA', 'Country_JAM', 'Country_JOR', 'Country_JPN', 'Country_KAZ', 'Country_KEN', 'Country_KGZ', 'Country_KHM', 'Country_KOR', 'Country_KWT', 'Country_LAC', 'Country_LAO', 'Country_LBN', 'Country_LBR', 'Country_LBY', 'Country_LCA', 'Country_LCN', 'Country_LDC', 'Country_LIC', 'Country_LKA', 'Country_LMC', 'Country_LMY', 'Country_LSO', 'Country_LTE', 'Country_LTU', 'Country_LUX', 'Country_LVA', 'Country_MAC', 'Country_MAR', 'Country_MDA', 'Country_MDG', 'Country_MDV', 'Country_MEA', 'Country_MEX', 'Country_MIC', 'Country_MKD', 'Country_MLI', 'Country_MLT', 'Country_MMR', 'Country_MNA', 'Country_MNE', 'Country_MNG', 'Country_MOZ', 'Country_MRT', 'Country_MUS', 'Country_MWI', 'Country_MYS', 'Country_NAC', 'Country_NAM', 'Country_NCL', 'Country_NER', 'Country_NGA', 'Country_NIC', 'Country_NLD', 'Country_NOR', 'Country_NPL', 'Country_NZL', 'Country_OED', 'Country_OMN', 'Country_OSS', 'Country_PAK', 'Country_PAN', 'Country_PER', 'Country_PHL', 'Country_PNG', 'Country_POL', 'Country_PRE', 'Country_PRI', 'Country_PRK', 'Country_PRT', 'Country_PRY', 'Country_PSE', 'Country_PSS', 'Country_PST', 'Country_PYF', 'Country_QAT', 'Country_ROU', 'Country_RUS', 'Country_RWA', 'Country_SAS', 'Country_SAU', 'Country_SDN', 'Country_SEN', 'Country_SGP', 'Country_SLB', 'Country_SLE', 'Country_SLV', 'Country_SOM', 'Country_SRB', 'Country_SSA', 'Country_SSD', 'Country_SSF', 'Country_SST', 'Country_STP', 'Country_SUR', 'Country_SVK', 'Country_SVN', 'Country_SWE', 'Country_SWZ', 'Country_SYR', 'Country_TCD', 'Country_TEA', 'Country_TEC', 'Country_TGO', 'Country_THA', 'Country_TJK', 'Country_TKM', 'Country_TLA', 'Country_TLS', 'Country_TMN', 'Country_TON', 'Country_TSA', 'Country_TSS', 'Country_TTO', 'Country_TUN', 'Country_TUR', 'Country_TZA', 'Country_UGA', 'Country_UKR', 'Country_UMC', 'Country_URY', 'Country_USA', 'Country_UZB', 'Country_VCT', 'Country_VEN', 'Country_VIR', 'Country_VNM', 'Country_VUT', 'Country_WLD', 'Country_WSM', 'Country_YEM', 'Country_ZAF', 'Country_ZMB', 'Country_ZWE']
                      Country Name  Year     Lag_1     Lag_2     Lag_3    Target  \
    0  Africa Eastern and Southern  1994  8.266327  8.270724  8.179629  8.138291   
    1  Africa Eastern and Southern  1995  8.138291  8.266327  8.270724  7.908446   
    2  Africa Eastern and Southern  1996  7.908446  8.138291  8.266327  7.823908   
    3  Africa Eastern and Southern  1997  7.823908  7.908446  8.138291  7.783654   
    4  Africa Eastern and Southern  1998  7.783654  7.823908  7.908446  7.812734   
    
       Country_AFE  Country_AFG  Country_AFW  Country_AGO  ...  Country_VEN  \
    0         True        False        False        False  ...        False   
    1         True        False        False        False  ...        False   
    2         True        False        False        False  ...        False   
    3         True        False        False        False  ...        False   
    4         True        False        False        False  ...        False   
    
       Country_VIR  Country_VNM  Country_VUT  Country_WLD  Country_WSM  \
    0        False        False        False        False        False   
    1        False        False        False        False        False   
    2        False        False        False        False        False   
    3        False        False        False        False        False   
    4        False        False        False        False        False   
    
       Country_YEM  Country_ZAF  Country_ZMB  Country_ZWE  
    0        False        False        False        False  
    1        False        False        False        False  
    2        False        False        False        False  
    3        False        False        False        False  
    4        False        False        False        False  
    
    [5 rows x 241 columns]
    

##### 4.3 Feature Scaling

Linear Regression and XGBoost,to some extent are sensitive to feature scales,while Random Forest is less affected.

Scaling standardizes the features so that they have:

- Mean = 0

- Standard deviation = 1

This puts all features on an equal footing, so models can learn more effectively.


```python
from sklearn.preprocessing import StandardScaler

# Define feature_cols if not already defined
feature_cols = [col for col in lagged_df.columns if col.startswith('Lag_') or col.startswith('Country_')]

scaler = StandardScaler()
X_scaled_df = pd.DataFrame(
    scaler.fit_transform(lagged_df[feature_cols]),
    columns=feature_cols
)
# Check mean and standard deviation after scaling
print("Feature means after scaling:\n", X_scaled_df.mean())
print("\nFeature standard deviations after scaling:\n", X_scaled_df.std())
X_scaled_df.head()

```

    Feature means after scaling:
     Lag_1          1.170420e-17
    Lag_2         -4.486612e-17
    Lag_3         -6.047172e-17
    Country_AFE    1.170420e-17
    Country_AFG    1.560561e-17
                       ...     
    Country_WSM    7.802803e-18
    Country_YEM    1.950701e-17
    Country_ZAF    2.340841e-17
    Country_ZMB    2.340841e-17
    Country_ZWE    3.901401e-17
    Length: 238, dtype: float64
    
    Feature standard deviations after scaling:
     Lag_1          1.000069
    Lag_2          1.000069
    Lag_3          1.000069
    Country_AFE    1.000069
    Country_AFG    1.000069
                     ...   
    Country_WSM    1.000069
    Country_YEM    1.000069
    Country_ZAF    1.000069
    Country_ZMB    1.000069
    Country_ZWE    1.000069
    Length: 238, dtype: float64
    




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
      <th>Lag_1</th>
      <th>Lag_2</th>
      <th>Lag_3</th>
      <th>Country_AFE</th>
      <th>Country_AFG</th>
      <th>Country_AFW</th>
      <th>Country_AGO</th>
      <th>Country_ALB</th>
      <th>Country_ARB</th>
      <th>Country_ARE</th>
      <th>...</th>
      <th>Country_VEN</th>
      <th>Country_VIR</th>
      <th>Country_VNM</th>
      <th>Country_VUT</th>
      <th>Country_WLD</th>
      <th>Country_WSM</th>
      <th>Country_YEM</th>
      <th>Country_ZAF</th>
      <th>Country_ZMB</th>
      <th>Country_ZWE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.118645</td>
      <td>0.111994</td>
      <td>0.089759</td>
      <td>15.297059</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>...</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.092044</td>
      <td>0.111086</td>
      <td>0.108475</td>
      <td>15.297059</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>...</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.044291</td>
      <td>0.084653</td>
      <td>0.107571</td>
      <td>15.297059</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>...</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.026727</td>
      <td>0.037200</td>
      <td>0.081265</td>
      <td>15.297059</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>...</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.018364</td>
      <td>0.019747</td>
      <td>0.034042</td>
      <td>15.297059</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>...</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
      <td>-0.065372</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 238 columns</p>
</div>



##### 4.4 Train-Test Split

Splitting the dataset into training and testing sets ensures that we can train the model on one portion of the data and evaluate its performance on unseen data.
>
For time-series or forecasting problems, it’s important to use temporal splitting so that the model is tested on future periods only, avoiding data leakage.


```python

from sklearn.model_selection import train_test_split

y = lagged_df['Target']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Optionally, split based on year to ensure temporal separation
train_df = lagged_df[lagged_df['Year'] < 2020]
test_df = lagged_df[lagged_df['Year'] >= 2020]
X_train = train_df[feature_cols]
y_train = train_df['Target']
X_test = test_df[feature_cols]
y_test = test_df['Target']

# Scale the features for train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Check the size of each split


```python
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

```

    Train shape: (6110, 238)
    Test shape: (1175, 238)
    

Confirm temporal separation


```python
print("Train years:", train_df['Year'].min(), "-", train_df['Year'].max())
print("Test years:", test_df['Year'].min(), "-", test_df['Year'].max())

```

    Train years: 1994 - 2019
    Test years: 2020 - 2024
    

Check row overlaps for data leakage prevention


```python
train_ids = set(train_df.index)
test_ids = set(test_df.index)
print("Overlap:", len(train_ids.intersection(test_ids)))

```

    Overlap: 0
    

Verification of the x_train and x_test


```python
print(train_df['Year'].unique())


```

    [1994 1995 1996 1997 1998 1999 2000 2001 2002 2003 2004 2005 2006 2007
     2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019]
    


```python
print(test_df['Year'].unique())
```

    [2020 2021 2022 2023 2024]
    
