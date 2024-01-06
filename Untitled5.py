#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Data Cleaning and Preperation


# In[2]:


#80 percent of the analyst time is spent upn loading, cleaning and transformation and rearranging. 


# In[3]:


#Pandas in python is a tool that enables you to manupulate data in the right form. 


# In[4]:


#In this Chapter:
#missing data
#duplicate data 
#string manupulation
#some other analytical data transformation


# In[5]:


#Handling Missing data.
#There may be missing data in our list. We can solve this issue of missing data using the python pandas library.


# In[7]:


#First for our use we create a series with a mssing data in it. 
import pandas as pd
import numpy as np
series1=pd.Series([1,2,3,np.nan,0])
series1


# In[8]:


#you can use is na to check the boolean expression giving the presence or non presence of null value in the given series
series1.isna()
#This convetion of referring to missing data as NA is derived from the R programming language
#In statistics the data that is not aviable indicates two things, 1 is either the data is not present in the given list, 2 is that the data was not observed
#When cleaning up the data it is necessary to the data analysis with the missing data itself beacuse the missing data may be a factor for problems or potential biases in the data.


# In[11]:


#The built in python none value is also treated as null value in the python programming language. 
series2=pd.Series([1,2,3,np.nan,None,"hardwork"])
series2.isna()
#You can observe that the result indicates the none value also as the non value.


# In[13]:


#There are some of the methods regarding na that you can use to check the presence of null values in the given data.
#dropna : Detects Null values by: Filter axis lables, can give threshold on how much missing data to include.
#fillna: fills in missing data with some value or usinf an interpolation method such as ffil
#isna: gives the boolean expression to indicate wether the given expression is true or false.
#notna: this is particularly a negetiation statement,this gives true if the element that we are checking is not na.


# In[15]:


#drop na is helpful in drpping the missing data out of the data frame or series.
#lets look at the series 2 first
series2


# In[3]:


import pandas as pd
import numpy as np
series2=pd.Series([1,2,3,np.nan,None,"hardwork"])
series2.dropna()


# In[6]:


#You can observe that the null part is dropped. In the given list.
#sometimes the same thing can be done as 
series2[series2.notna()]


# In[13]:


#Want to deal with these things in the data frame, you can try as follows
data1=pd.DataFrame([[1,np.nan,3],[None, np.nan, 1],[np.nan,np.nan,np.nan],[np.nan,np.nan,2]])
data1


# In[14]:


#Want to drop all rows that contains the null values we can workout as
data1.dropna()


# In[15]:


#if you want to specify how you want to drop the null values you can try as follows
data1.dropna(how="all")
#you can see that in the result only the row that contained only null values are dropped out of the rows.


# In[18]:


#want to drop the null values from the column. You can try as
data1.dropna(axis="columns", how="all")


# In[23]:


data2=pd.DataFrame(np.random.standard_normal((7,3)))
data2.iloc[:4,1]=np.nan#this method of slicing i think needs explanation, because as a begineer you will alwyas get confused in the slicing stuff.
data2.iloc[:2,2]=np.nan
data2
#here the slicing is particularly done, on specifying one column at a time on a row. I hope this explanation will justify the slicing
#if you are confused about slicing, you can check our first series.


# In[25]:


#In this you can specify threshold to specify, to remove the rows from the data frame.
data2.dropna(thresh=2)#these removes the rows having 2 or more than 2 null values


# In[29]:


#You can also fill the missing filed with the fillna method 
data2.fillna(data2.mean(), limit=2)#This code operates to fill the missing value in the given data frame with the mean of the all elements in the data frame.
#The limit is specified here to fill the maximum number of null values. 


# In[39]:


#You can try other things like ffill or bfill, these are forward fill and backfill
#In order to use them you can specify them as method. 
data2.fillna(method="bfill") # this method of backfill works on columns
#in this ffill doesnot work but why?, i want to find the answer of this question, it checks wether you understood the code or not.


# In[40]:


#Data Transformation


# In[44]:


#Some times your data may contain the duplicated data int he seires. You can remove the duplicated data using the duplicated data.4
data3=pd.DataFrame({"a1":["a","b"]*3+["c"],
                   "a2":[1,2,2,2,3,3,4]})
data3


# In[46]:


#You can see that in a value are repeated.
data3.duplicated()


# In[47]:


#To drop the values from the duplicated.
data3.drop_duplicates()
#You can see that the value that is repeated here is dropped using the drop command.


# In[50]:


#You want to, drop the duplicate values based on only one column than you can proceed as.
data3.drop_duplicates(subset=["a1"])
#In the result you can observe that only the duplicated elements of column1 are eliminted. By this the data frame is presented as below.


# In[55]:


#We can use keep="last" in order to include the last repeated value in the data this process can proceed as follows
data3.drop_duplicates(["a1","a2"],keep="last")


# In[2]:


#Transforming the data using a function or mapping.
#Map Method
#I think this method this method is important.
#In this method you can arrange messed up data with your data if they have the same key. 
#You need to create a dictionary in order to do this. 
#lets first create a data frame
import pandas as pd 
import numpy as np
data4 = pd.DataFrame({"animal": ["dog1", "dog2", "dog2",
                                  "dog3", "dog4", "dog5",
                                  "dog6", "dog7", "dog8"],
                         "number": [2, 3, 10, 6, 7, 9, 6, 5, 6]})
data4


# In[9]:


#Suppose you wanted to add a column that contains, name of these dogs
namesofanimal={
    "dog3":"rocky",
    "dog1":"max",
    "dog2":"jenny",
    "dog4":"micky",
    "dog5":"jenifer",
    "dog6":"loud",
    "dog7":"python",
    "dog8":"java"
    #giving name of dogs with the name of programming language is really a good idea, you know it benifits a lot.
}


# In[11]:


data4["names"]=data4["animal"].map(namesofanimal)
data4


# In[12]:


#You can observe how these corresponding names are defined to the animal. 


# In[13]:


#Replacing values
#The process of replacing values can simply be done by the replace method. 
data5=pd.Series([1,2,3,4,5,6])
data5


# In[15]:


#want to replace 3 with null value and 1 with 3 
#you can write
data5.replace({1:3,3:np.nan})#from this way by writing a dictionary you can specify and replace the values in the given series.


# In[49]:


#Want to modify the index in the data frame you can try
#first lets create a data frame
data6=pd.DataFrame(np.arange(12).reshape((4,3)),
                   index=["January","February","March","April"],
                   columns=["a","b","c"])
def function1(x):
    return x[:3].upper()
data7=data6.index.map(function1)
data7


# In[50]:


data6.index=data6.index.map(function1)


# In[52]:


data6


# In[57]:


test=data6.index
test


# In[58]:


#if you want to just change the values in the data set without modifying the original one, you can try for the expression as
data6.rename(index=str.title, columns=str.upper)


# In[65]:


#you can also use, dictionary to change the index names
data6.rename(index={"Jan":"Mon"},
           columns={"A":"a"})


# In[63]:


#Discretization and Binning


# In[67]:


#One of the ways of the categorization of the data is using the cut method. 
#lets consider a list of ages of the people
ages=[20,21,23,45,70,78]
category=[18,20,30,40,50,70,80]
category_1=pd.cut(ages, category)
category_1


# In[68]:


#you can look at numerical code od these categories, these values are interval value types containing the lower and upper limit of each bin
category_1.codes


# In[69]:


#Want to just look at the categories
category_1.categories


# In[70]:


#want to look at the number of elements present in the each category
pd.value_counts(category_1)


# In[73]:


#While expressing the cateogires you can see that, either of the sides are closed in the expression
#You can specify which side is closed and which side is open using the boolean expression as
pd.cut(ages, category, right=False)


# In[81]:


#You can make the cateogory simple by naming them
#The method of naming these categories is called as labeling. 
category_names=["youth","youthadult","middleage1","middleage2","old","veryold","hospitalized"]
pd.cut(ages, category, labels=category_names)


# In[87]:


#Here you have given the python the cateogory you want to include.
#But if you just want to categorize the data you can try for. 
#lets firs create a data
data7=np.random.uniform(size=20)
result1=pd.cut(data7, 4, precision=2)#precision 2 means the decimal digits after the integer value is only in the given data


# In[88]:


pd.value_counts(result1)


# In[89]:


#Here each category has different number of values assigned to it.


# In[91]:


#A similar process done by the q cut helps to assign each cateogry and equal number of values
result2=pd.qcut(data7, 4, precision=2)
pd.value_counts(result2)


# In[ ]:


#You can observe the values are equal in each categories. This process is specially essential in dividing the data into quartiles.


# In[ ]:


#Detectinf and Filtering Outliers


# In[92]:


#Filtering or transforming outliers is performing array operation. In these scenarios you should understand the logic behind the operation


# In[93]:


#to understand this let's first create a dataset. 
data8=pd.DataFrame(np.random.standard_normal((1000,4)))
data8.describe()


# In[94]:


#You can ectract data from a specific column of the original data and set the codition to extract the data
col=data8[2]
col[col.abs()>2]#This selects those data in the column 3 of the dataframe data 8 that is freater than3.


# In[ ]:


#You can see that it has given us a huge list of the data. With all whose absolute value is greater than 2.


# In[96]:


#To select all the rows of the dataframe whose absolute value is greater than two. 
#You can use the any method.
data8[(data8.abs()>2).any(axis="columns")]


# In[ ]:


#You can observe that all the rows have at least one element that is greater than 2.
#We have used a paranthesis around the data8 to use the any method in here. 


# In[98]:


#sign is a method that is used in python to give the values the 1, -1 values
#for example in our original expression data, if we want to check the negative and positive structure of the data than we can use the code as.
np.sign(data8).head()


# In[99]:


#you can see that the elements presents first five rows of the dataframe mentioning the structre of data that explains the presence of positive and negative elements on the dataframe.


# In[103]:


#Permutation and Random Sampling.
# you can use pemutation to generate random numbers withing a specified limit
data9=pd.DataFrame(np.arange(35).reshape((5,7)))
array1=np.random.permutation(5)
array1


# In[104]:


#you can use this randomly generated permutation to use in the indexing the dataframe
#for this you can try using take with the name of the array and dataframe, or simply you can use iloc in the similar way
data9.take(array1)
#remember the ascending order is itself managed by the code. We don't need to manage the ascending order in this case.


# In[106]:


#We can also use, 
data9.iloc[array1]


# In[108]:


#Wr can also permut the columns using the same method. 
#In order to do this you need to define the axis here. 
data9.take(array1, axis="columns")


# In[110]:


#want to select a random subset from the data frame you can use the expression. 
data9.sample(n=3)#n=3 here specifies to select the 3 rows from the data frame. 


# In[111]:


#to allow slecting the repeat choices you can apply the command. 
data9.sample(n=10, replace=True)


# In[ ]:


#you can get those values that are free from the negation of repeat chouces.


# In[6]:


import pandas as pd
import numpy as np
#Computer Indicator and Dummy variables. 
#You can make a random dummy data frome from the values of a column
data9=pd.DataFrame(np.arange(35).reshape((5,7)))
data9


# In[7]:


pd.get_dummies(data9[1], dtype=float)
#datatype is indicated float to get the value in the numerical form. 


# In[34]:


#Supppose you have a group of strings, that are seperated by symbol | in order to create a dummy data frame out of this string you can use the inbuilt str.get_dummies method
#This method is particular designed to create a dummy dataframe from the given data frame. 
#you can look at the following data table. I have attached it at the, github repository.
data1=pd.read_table("C:\\Users\\user\\Desktop\\data1.dat",sep=",",engine="python")
data1


# In[37]:


dummies=data1["book_genres"].str.get_dummies("|")
dummies


# In[38]:


#You can see how the data frame is created, that is based on boolean, and the name of the columns are the names of book_genres. 


# In[40]:


#You can look at the dataset and the determine, join the dummy data set, witht the data2
newdataframe=data1.join(dummies.add_prefix("_Genre"))
newdataframe


# In[41]:


#Want to view the first row of the given dataframe. You can apply as follwos:
newdataframe.iloc[0]


# In[ ]:


#For conducting large data, this method, is not speedy. 
#It is better to deal with numpy array that wrap the result in a DataFrame. 


# In[46]:


#Want to create a dummy dataframe that contains the boolean expression that combines with panda.cut
values=np.arange(10)
values
bins=[0,2,4,6,8,10]
pd.get_dummies(pd.cut(values,bins),dtype="float64")


# In[59]:


#Extension Data Types
#dtype etension is used to define the type of data in the expression.
series1=pd.Series([1,2,3,None])
series1.dtype


# In[60]:


series1


# In[61]:


series1.isna()


# In[72]:


#Want to specify the data type at the first. 
series2=pd.Series([1,2,3,4,None], dtype=pd.StringDtype())
series2


# In[ ]:


#string arrays are mostly used in large datasets because they consume less memories


# In[73]:


#suppose you want to deal with the dataframe, and specify it's rows and columns as a different data frame you can proceed as
dataframe=pd.DataFrame({"a":[1,2,None,4],
                       "b":["one","tow","three","none"],
                       "c":[False,None,False,True]})
dataframe


# In[75]:


#now for example you want to, specify integer. 
dataframe["a"]=dataframe["a"].astype("Int64")
dataframe["b"]=dataframe["b"].astype("string")
dataframe["c"]=dataframe["c"].astype("boolean")
dataframe


# In[76]:


#You can refer to wes mickinney book to learn different data types that can be learend in this lesson. 


# In[ ]:


#String Manipulation
#One of the reasons python is so popular is beacuse of it's ability to manupulate strings. 


# In[78]:


#One of the ways to do it is to split the comma seperated strings into parts. 
string="a,b,c, d"
string.split(",")


# In[81]:


#split is often combined with strip to trim whitespace including the line breaks. 
trimmed=[x.strip() for x in string.split(",")]
trimmed


# In[82]:


#The strings can be again joined together with a certain symbol using the join method
";".join(trimmed)


# In[ ]:


#You can locate substrings inside the string using the method of index or find.
trimmed.index(";")
#or you can also try 
trimmed.find(";")
#Remember that, index method gives error if the string desried is not present int the given series.
#But find method gives the negative value if the string is not present in the given series.


# In[ ]:


#Another method that can be used with strings is called as replace method. 
#want to replave ; with , than you can try
trimmed.replace(";",",")
#Or want to replace with space than you can write. 
trimmed.replace(";"," ")


# In[2]:


# #Want to know about some methods in split that is used. These are python built in methods.
# count: retrn values of non overlapping occurences of substrings in the string.
# endswith: gives true if the string ends with the specified with given suffix
# startswith: gives tue if the string ends with prefix
# join: string delimiter for concatenating a sequence of other strings.
# index: return starting index of the first occurence of passed substring found in the string. If the given element is not found in the given string than it raises the exception as error.
# find: return position of first character of first occurrence of substring in the string like index but returns -1 if not found.
# rfind:retuen position of first character of last occurence of substring in the string. returns -1 if not found.
# replace: replace occurences of strin with another string
# strip, rstrip, lstrip: strip whitespace, r means trim on the right and l means trim on the left.
# split: break string into list of substrings using passed delimiter.
# lower: convert alphabet character to lowercase
# upper: convert aplphabet into upper case. 
# casefold: conver characters to lowercase and convert any region specific varibales character combinantion to a common comaprable form
# ljust, rjust: left justify and right justify, pd oppsite side of string with spaces, to return a string with minimum width.


# In[3]:


import pandas as pd
import numpy as np
#Regular expressions
#regular expression provide a flexible way to search or match string patterns in a text. It is primarily divided into three categories:
#1. Pattern Matching
#2. Substitution
#3. Splitting. 
#Lets look at one example and make this concept more clear. 


# In[9]:


import re 
#in this example we will describe text having more than one or more spaces using the regex method
string1="kali ubuntu\t  windows \tarch"
re.split(r"\s+", string1)


# In[11]:


#you can also use regex with the prbuilt strategy where you define the values seperator, and everytime you call regex you can use this method. 
regexmethod=re.compile(r"\s+")
regexmethod.split(string1)#whenever you call the regex method this process functions and thus we can get the corresponding results


# In[12]:


#to fund all the pattern that is matching the regex you can use the findall method
regexmethod.findall(string1)


# In[13]:


#similar process are search and match 
#search method delivers the first result matching the result
regexmethod.search(string1)


# In[17]:


#match method checks wether if the given condition is present at the intial part of the string. 
print(regexmethod.match(string1))
#since the condition specified is not met in the first part of the string. 


# In[18]:


#method sub will return a new string with occurences of the pattern replaced by a new string. 


# In[20]:


#for wrting regular expressions you should learn it seperately. 
#this course covers the application of regular expression. 
#for continiuing this course writing regular expression is not necessary.
#As you proceed through the course I want you to learn regular expression parallely.


# In[30]:


#String Functions in Pandas
#lets first make a dictionary than we will convert that dictionary into a series. 
dict1={"Dave": "dave@google.com", "Steve": "steve@gmail.com",
          "Rob": "rob@gmail.com", "Wes": np.nan}
series3=pd.Series(dict1)
series3


# In[31]:


#to check the null values
series3.isna()


# In[33]:


#to check wether certain string is contained in the table or not. We can use the expression
series3.str.contains("gmail")


# In[34]:


#want to change the data types
changed=series3.astype("string")
changed


# In[ ]:


#We have already discussed enough about extension data types.


# In[37]:


pattern=r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
changed.str.findall(pattern, flags=re.IGNORECASE)
#The ingnore case is used to make the core case insensitive.


# In[40]:


#str[index] and str.get(index) is used to extract vectorized element.
data10=changed.str.findall(pattern, flags=re.IGNORECASE).str[0]
data10


# In[42]:


data10.str.get(1)


# In[43]:


#You can also extract the data into dataframe by sing the method called as extract:
series3.str.extract(pattern, flags=re.IGNORECASE)


# In[44]:


#Categorical Data:
#This section introduces the pandas categorical type.. These tools will help you understand categorical data in statistics and machine learning application. 


# In[45]:


#Background and Motivation:
#first lets create a series and try to understand this. 
series4=pd.Series(["apple","orange","apple","apple"]*2)#*2 basically means type the list two times.
series4


# In[47]:


#i think upto this point you may have learned how to count the values
pd.value_counts(series4)


# In[48]:


#When handling these repeated datas, it will be a lot of mess if we leave them in the string form.
#It is better to convert the given set of data into the integer form for better handlin the process.
integerdata=pd.Series([0,1,0,0]*2)
reference=pd.Series(["apple","orange"])
integerdata


# In[49]:


reference


# In[50]:


# We can take the values and create the original series again. 
reference.take(integerdata)


# In[ ]:


#The array of distinct values can be called the categories, dictionary, or levels of the data. 
#The integer value categories are called category codes or simply codes. 


# In[54]:


fruits1 = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits1)
rng = np.random.default_rng(seed=12345)
data11 = pd.DataFrame({'fruit': fruits1,
                       'basket_id': np.arange(N),
                      'count': rng.integers(3, 15, size=N),
                       'weight': rng.uniform(0, 4, size=N)},
                      columns=['basket_id', 'fruit', 'count', 'weight'])


# In[55]:


data11


# In[58]:


#Want to extract certain column from the data frame and name them as category you can try using the code
categoryformdataframe=data11["fruit"].astype('category')
categoryformdataframe

                                        


# In[59]:


#the values in categorydataframe are now the categorical values of panda and you can access them via the array attribute
array1=categoryformdataframe.array
array1


# In[60]:


#you can see the categories inside these arrays by writing categories. 
array1.categories


# In[62]:


#you can see the integer code format using the code
array1.codes


# In[63]:


#you can also directly create categories from a list
my_categories=pd.Categorical(["a","b","b","a","c","c","d","e","e","f"])
my_categories


# In[65]:


#if you have the codes and the categories seperated you can integrate the categories with the codes using the from_codes
categories=["a","b"]
codes1=[1,0,0,1,0,1]
datanow=pd.Categorical.from_codes(codes1, categories)
datanow


# In[66]:


#normally the generated data is ordered on the basis of the index names.
#some times the ordere may not be correct at this case you can sepcify the order using ordered=True
#for an unordered data you can use the process
datanow.as_ordered()


# In[70]:


#Computation using categories. 
#Computation in categories behave the same way as the arrays are computed for this process we can proceed as follows
rng=np.random.default_rng(seed=12345)
randomdata=rng.standard_normal(1000)
randomdata
#suppose you want to have the categorization, you can use qcuts
new=pd.qcut(randomdata, 4)
new


# In[72]:


#now this kind of data is less useful, it would be more efficient from reading point of view if we had defined the labels here
new1=pd.qcut(randomdata, 4, labels=["q1","q2","q3","q4"])
new1


# In[74]:


#you can also use the integer format by using the command code
new1.codes


# In[82]:


#we can use the grupby command to gain more statistical data in this case. 
results2 = (pd.Series(randomdata)
               .groupby(new1)
               .agg(['count', 'min', 'max'])
            .reset_index())
results2
#then you can extract particular information from the data frame by the methods that we have already learned
results2["index"]


# In[83]:


#Putting the data on the categories can improve the memory. 
#To learn about memory details and categories you can look at the book by wes mickinney.


# In[ ]:


#You can perform different actions with categories that we had learned previously in series also.


# In[ ]:


# #here are some of the important category things that you can learn: This table is taken from the book by wes mickinney
# add_categories:Append new (unused) categories at end of existing categories
# as_ordered:Make categories ordered
# as_unordered:Make categories unordered
# remove_categories:Remove categories, setting any removed values to null
# remove_unused_categories:Remove any category values that do not appear in the data
# rename_categories:Replace categories with indicated set of new category names; cannot change the number of categories
# reorder_categories:Behaves like rename_categories, but can also change the result to have ordered categories
# set_categories:Replace the categories with the indicated set of new categories; can add or remove categories


# In[ ]:


#Here ends our journey od data cleaning and preperation. Now we will deal with data wrangling and joining shapes
#I highly recommend you to look at all the series before moving on to the next chapter.
#If you have aready done it hats off to your commitment. 


# In[ ]:


#Regards
#mechengics
#Ankit Sangroula.

