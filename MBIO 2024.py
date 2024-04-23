#Project Name:MbioHoward
#Date created:2/4/24
#Last Edited:4/16/24
#Author: Ademola Abdulkadir

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from statsmodels.formula.api import ols

#from google.colab import drive
#drive.mount('/content/drive')

#/content/faostat_flags.csv
#/content/faostat_country_codes.csv
#/content/faostat_crops.csv
#/content/faostat_units.csv
#/content/upov_members.csv

#Loading all the data sets
os.getcwd()
os.chdir('/Users/ademo/Documents/Howard 2024/Social Impact')
flagDf = pd.read_csv('faostat_flags.csv')
countryDf = pd.read_csv('faostat_country_codes.csv')
cropsDf = pd.read_csv('faostat_crops2.csv', encoding= 'cp1252')
unitsDf = pd.read_csv('faostat_units.csv')
membersDf = pd.read_csv('upov_members.csv')

flagDf.head()

countryDf.head()

#Cleaning the crops data set so that only only the observations with element = area harvested or production remain
cropsDf.head()
mask = (cropsDf['Element'] == 'Area harvested') | (cropsDf['Element'] =='Production')
cropsDf = cropsDf[mask]
cropsDf = cropsDf.drop(['Area Code', 'Area Code (M49)', 'Item Code', 'Item Code (CPC)', 'Element Code'], axis = 1)
print(cropsDf['Area'].unique())
cropsDf.head()

unitsDf.head()

membersDf.head()

dropColF = [col for col in cropsDf.columns if 'F' in col]
dropColN  = [col for col in cropsDf.columns if 'N' in col]
print(dropColF)
print(dropColN)

#Remvoing the year colums that hold the quality control flag and notes about the quality control flag
cropsDf = cropsDf.drop(dropColF, axis = 1)
cropsDf = cropsDf.drop(dropColN, axis = 1)
cropsDf.columns = cropsDf.columns.str.strip('Y')


#Needs to be cleaned up
#isolating the year columns for reference 
yearCol = cropsDf.columns
yearCol = yearCol.drop('Area')
yearCol = yearCol.drop('Item')
yearCol = yearCol.drop('Element')
yearCol = yearCol.drop('Unit')
yearCol

#Creating one year column rather than having a column for each year
cropsDf = pd.melt(cropsDf, id_vars = ['Item', 'Area', 'Element', 'Unit'], value_vars = yearCol, var_name = 'Years', value_name = 'Quantity')

#Group countries by continent

africa = ['Sudan', 'South Africa', 'Nigeria','Chad', 'Niger', 'Angola', 'Somalia', 'Mali', 'Mozambique', 'Algeria',
          'Madagascar', 'Burundi','Ethiopia', 'Egypt', 'Central African Republic', 'Tanzaia','Kenya',
          'Uganda', 'Algeria', 'Morroco', 'Ghana', "Côte d'Ivoire", 'Cameroon', 'Mali', 'Burkina Faso', 'Malawi',
          'Zambia', 'Libya', 'Senegal', 'Senegal', 'Zimbabwe', 'Guninea', 'Rwanda','Benin',' Burundi', 'Tunisia',
          'South Sudan', 'Togo', 'Sierra Leone', 'Liberia', 'Mauritania', 'Eritrea', 'Gambia', 'Botswana', 'Namibia',
          'Gabon', 'Lesotho', 'Guinea-Bissau','Equatorial Guinea', 'Mauritius', 'Eswatini', 'Dijibouti', 'Comoros',
          '52	Cabo Verde', 'Sao Tome and Principe', 'Seychelles' ]

europe = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia',
          'Cyprus', 'Czech', 'Denmark', 'European Union, Estonia', 'Finland', 'France', 'Germany', 'Greece',
          'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta',
          'Moldova', 'Monaco', 'Montenegro', 'Netherlands', 'North Macedonia', 'Norway', 'Poland',  'Portugal',
          'Romania', 'Russia', 'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden' , 'Switzerland', 'Ukraine','United Kingdom']

europeDf = europeDf = cropsDf[cropsDf['Area'].isin(europe)]


eastAfrica = ['Somalia', 'Ethiopia', 'United Republic of Tanzania', 'Mozambique', 'South Sudan']
upov = ['Egypt', 'Ghana', 'Kenya', 'Morroco', 'South Africa', 'United Republic of Tanzania']

eastAfricaDf = cropsDf[cropsDf['Area'].isin(eastAfrica)]
upovDf = cropsDf[cropsDf['Area'].isin(upov)]
africaDf = cropsDf[cropsDf['Area'].isin(africa)]

#Creating a data for each country to make regression analysis easier
somDf = eastAfricaDf[(eastAfricaDf['Area'] == 'Somalia') & (eastAfricaDf['Element'] == 'Production')]
ethDf = eastAfricaDf[(eastAfricaDf['Area'] == 'Ethiopia') & (eastAfricaDf['Element'] == 'Production')]
tanDf = eastAfricaDf[(eastAfricaDf['Area'] == 'United Republic of Tanzania') & (eastAfricaDf['Element'] == 'Production')]
mozDf = eastAfricaDf[(eastAfricaDf['Area'] == 'Mozambique') & (eastAfricaDf['Element'] == 'Production')]
sudDf = eastAfricaDf[(eastAfricaDf['Area'] == 'South Sudan') & (eastAfricaDf['Element'] == 'Production')]

safDf = upovDf[(upovDf['Area'] == 'South Africa') & (upovDf['Element'] == 'Production')]
morDf = upovDf[(upovDf['Area'] == 'Morroco') & (upovDf['Element'] == 'Production')]
kenDf = upovDf[(upovDf['Area'] == 'Kenya') & (upovDf['Element'] == 'Production')]
ghaDf = upovDf[(upovDf['Area'] == 'Ghana') & (upovDf['Element'] == 'Production')]
egyDf = upovDf[(upovDf['Area'] == 'Egypt') & (upovDf['Element'] == 'Production')]




#Removiing the leading character of the year column strings
yearCol = yearCol.str.replace('Y', '', regex = True)

cropCnt = cropsDf.drop('Area', axis = 1)
cropCnt = cropCnt[cropCnt['Element'] == 'Production']
cropCnt






test = ethDf.sample(10)
y = ethDf.loc[ethDf['Item'] == 'Bananas', 'Quantity']
#x = ethDf.loc[ethDf['Item'] == 'Bananas', 'Years']
x = ethDf['Years']
plt.scatter(test['Years'], test['Item'] )
plt.xticks(rotation=70)

#Years used to populate the dummy variable "Membership" based on when countries
# became UPOV members. Membership is a categorical variable
upovDf['Years'] = pd.to_numeric(upovDf['Years'], errors = 'coerce')
membership = []
for i, row in upovDf.iterrows():
    if (row['Area'] == 'Egypt') & (row['Years'] < 2019):
        membership.append(0)
    if (row['Area'] == 'Egypt') & (row['Years'] >= 2019):
        membership.append(1)
    if (row['Area'] == 'Ghana') & (row['Years'] < 2021):
        membership.append(0)
    if (row['Area'] == 'Ghana') & (row['Years'] >= 2021):
        membership.append(1)
    if (row['Area'] == 'Kenya') & (row['Years'] < 1999):
        membership.append(0)
    if (row['Area'] == 'Kenya') & (row['Years'] >= 1999):
        membership.append(1)       
    if (row['Area'] == 'Morroco') & (row['Years'] < 2006):
        membership.append(0)
    if (row['Area'] == 'Morroco') & (row['Years'] >= 2006):
        membership.append(1)    
    if (row['Area'] == 'South Africa') & (row['Years'] < 1977):
        membership.append(0)
    if (row['Area'] == 'South Africa') & (row['Years'] >= 1977):
        membership.append(1)
    if (row['Area'] == 'United Republic of Tanzania') & (row['Years'] < 2015):
        membership.append(0)
    if (row['Area'] == 'United Republic of Tanzania') & (row['Years'] >= 2015):
        membership.append(1)
        
#This dataframe now has a membership column that expresses the categorical
# numberically        
upovDf['Membership'] = membership

#Mulitvariate regression with area harvested and production as dependent 
#variables. Year and membership are independetn variables
df = upovDf[upovDf['Element'] == 'Production']


df.fillna(df.mean(), inplace = True)

X = df[['Membership', 'Years' ]]
Y = df['Quantity']

regr = linear_model.LinearRegression()
regr.fit(X, Y)

reg = ols('Quantity ~ Membership', df).fit()
reg.summary()

#df2 has the area harvested values
df2 = upovDf[upovDf['Element'] == 'Area harvested']
df2.fillna(df2.mean(), inplace = True)

X2 = df2[['Membership', 'Years' ]]
Y2 = df2['Quantity']

regr2 = linear_model.LinearRegression()
regr2.fit(X2, Y2)

reg2 = ols('Quantity ~ Membership', df2).fit()
reg2.summary()