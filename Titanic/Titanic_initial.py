
# coding: utf-8

# In[1]:

import pandas as pd


# In[2]:

df = pd.read_csv('/Users/GAJJAR/Desktop/ML/Kaggle/Titanic Machine Learning from Disaster/train.csv')


# In[119]:

features = df.copy()
features.head(100)


# In[64]:

features.drop('Survived', axis=1, inplace=True)


# In[65]:

features.drop('Name', axis=1, inplace=True)


# In[66]:

features.drop('Ticket', axis=1, inplace=True)


# In[67]:

features.drop('PassengerId', axis=1, inplace=True)


# In[68]:

features.drop('Cabin', axis=1, inplace=True)


# In[69]:

features.drop('Embarked', axis=1, inplace=True)


# In[70]:

features = features.fillna(0)


# In[9]:

from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[71]:

features = pd.get_dummies(features, columns=["Sex"])


# In[11]:

features = pd.get_dummies(features, columns=["Cabin", "Embarked"])


# In[72]:

features.head()


# In[73]:

out = df['Survived']


# In[74]:

from sklearn import tree


# In[75]:

clf = tree.DecisionTreeClassifier()


# In[76]:

clf.fit(features, out)


# In[77]:

clf.score(features, out)


# In[19]:

from sklearn.cross_validation import train_test_split


# In[78]:

x_train, x_test, y_train, y_test = train_test_split(features, out, test_size=0.4)


# In[79]:

clf.fit(x_train, y_train)


# In[80]:

clf.score(x_test, y_test)


# In[81]:

clf.score(x_train, y_train)


# In[45]:

# Get Test data


# In[82]:

df_test = pd.read_csv('/Users/GAJJAR/Desktop/ML/Kaggle/Titanic Machine Learning from Disaster/test.csv')


# In[83]:

df_test.head()


# In[84]:

features_test = df_test.copy()


# In[85]:

features_test.drop('Name', axis=1, inplace=True)


# In[86]:

features_test.drop('Ticket', axis=1, inplace=True)


# In[87]:

features_test.drop('PassengerId', axis=1, inplace=True)


# In[89]:

features_test.drop('Cabin', axis=1, inplace=True)


# In[90]:

features_test.drop('Embarked', axis=1, inplace=True)


# In[91]:

features_test = features_test.fillna(0)


# In[92]:

features_test = pd.get_dummies(features_test, columns=["Sex"])


# In[93]:

#features_test = pd.get_dummies(features_test, columns=["Cabin", "Embarked"])


# In[95]:

out_test = clf.predict(features_test)


# In[58]:

x_train.shape


# In[59]:

features_test.shape


# In[60]:

df_test.shape


# In[61]:

features_test.columns


# In[62]:

x_train.columns


# In[97]:

genarated_out = df_test.copy


# In[115]:

df_test['Survived'] = out_test


# In[104]:

df_test.columns


# In[103]:

df_test.drop('Name', axis=1, inplace=True)


# In[105]:

df_test.drop('Pclass', axis=1, inplace=True)


# In[106]:

df_test.drop('Sex', axis=1, inplace=True)


# In[107]:

df_test.drop('Age', axis=1, inplace=True)


# In[108]:

df_test.drop('SibSp', axis=1, inplace=True)


# In[109]:

df_test.drop('Parch', axis=1, inplace=True)


# In[110]:

df_test.drop('Ticket', axis=1, inplace=True)


# In[111]:

df_test.drop('Fare', axis=1, inplace=True)


# In[112]:

df_test.drop('Cabin', axis=1, inplace=True)


# In[113]:

df_test.drop('Embarked', axis=1, inplace=True)


# In[116]:

df_test.drop('Srvived', axis=1, inplace=True)


# In[118]:

df_test.to_csv('generated_out.csv')


# In[120]:

features.shape


# In[ ]:



