#!/usr/bin/env python
# coding: utf-8

# # Numpy

# 1) fundamental package for scientific computing with python 
# 2) homogenous multidimention array
# 3) numpy dimension are called axes
# 4) numpy's array class is ndarray also known as alias array

# # Installing Numpy
conda install numpy
# In[1]:


import numpy as np


# In[2]:


a = np.arrange(6)


# In[3]:


a


# # Array

# In[4]:


a = np.array([1,2,3,4,5])


# In[5]:


a

Two dimension array
# In[6]:


a =  np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])


# In[7]:


a


# In[8]:


a[0]


# In[9]:


a =  np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])


# In[10]:


a


# In[12]:


a = np.array(20).reshape(4,5)


# In[13]:


a


# In[14]:


a.ndim


# In[17]:


a = np.array(20).reshape(4, 5)


# In[18]:


a


# In[19]:


a.shape


# In[20]:


a.size


# In[21]:


a.dtype.name


# In[22]:


a.itemsize


# # Array Creation

# In[23]:


np.array([1,2,3,4])


# In[24]:


np.array([ [1,2], [3,4] ] , dtype=complex)


# In[25]:


np.array( [ [complex(1, 1), complex(1, 2)], [3,4]], dtype=complex)


# In[26]:


np.zeros( (3,4))


# In[27]:


np.ones( (3,4))


# In[28]:


np.empty( (3,4))


# In[29]:


np.empty((3,3))


# In[31]:


np.arange(10,30,5)


# In[32]:


np.linspace(0,2,9)


# In[33]:


x= np.ones(2,dtype=np.int64)


# In[34]:


x


# In[35]:


np.random.rand(2,3)


# In[36]:


a= np.array([1,2,3,4,6,5])


# In[37]:


np.sort(a)


# In[38]:


a=np.array([1,2,3,4])


# In[39]:


b=np.array([3,4,5,6])


# In[40]:


np.concatenate((a,b))


# In[41]:


a=np.arange(20)


# In[42]:


a


# In[43]:


b=a.reshape(4, 5)


# In[44]:


b


# In[45]:


b.shape


# In[46]:


a.ravel()


# In[47]:


a.resize(4,5) #return the array flaatend


# In[48]:


a


# # Add new axis  to array

# In[49]:


a= np.array([1,23,4,5])


# In[50]:


a.shape


# In[51]:


a2= a[np.newaxis,:]


# In[52]:


a2


# In[53]:


a2.shape


# In[54]:


row_vector=a[np.newaxis,:]


# In[55]:


row_vector


# In[56]:


col_vector=a[:,np.newaxis]


# In[57]:


col_vector


# # Basic operation

# In[58]:


data = np.array([1,2])


# In[59]:


ones = np.ones(2,dtype=int)


# In[60]:


data+ones


# In[61]:


data - ones


# In[62]:


data * data


# In[63]:


data.sum()


# In[65]:


b = np.array([[1,2],[3,4]])


# In[67]:


b.sum(axis=0)


# In[68]:


b.sum(axis=1)


# # Broadcasting

# In[69]:


data = np.array([1.0,2.0])


# In[70]:


data*1.6


# In[71]:


ones_row=np.array([[1,1]])


# In[72]:


data + ones_row


# In[7]:


10*np.cos(data)


# In[5]:


a=np.array([[1,2],[3,4]])


# In[8]:


a


# In[10]:


a.min()


# In[9]:


a.sum()


# In[11]:


a.max()


# In[12]:


np.sqrt(a)


# In[13]:


np.sin(a)


# In[14]:


np.cos(a)


# In[15]:


np.exp(a)


# In[16]:


a[0]


# In[17]:


a[-1]


# In[18]:


a[2:4]


# In[19]:


a[:4]


# In[20]:


a[0:5:2]=1000


# In[21]:


a


# In[24]:


data = np.array([ [1,2,3],[4,5,6],[4,6,7] ])


# In[25]:


data


# In[26]:


data[0,1]


# In[27]:


data[1:3]


# In[29]:


data[0:2:0]


# # Boolean Indexing

# In[30]:


a = np.array([ [1,2],[3,4],[4,5]])


# In[31]:


a


# In[32]:


bool_idx =(a>2)


# In[33]:


print(bool_idx)


# In[34]:


a[bool_idx]


# In[35]:


print(a[a>2])


# In[36]:


x= np.array([1.,-1.,-2.,3])   #to add a constent to allnegative element 


# In[37]:


x[x<0]+=20


# In[38]:


x


# In[39]:


a=np.array([[1,2,3,4],[3,4,5,6],[9,6,5,4]])


# In[40]:


b = np.nonzero(a<5)

you can use np.nonzeros to print the indices of element that are for exm. less than 5
# In[41]:


b


# In[42]:


a[b]


# In[43]:


c = (a>4)&(a==5)


# In[44]:


c


# In[45]:


c = (a>5) | (a==2)


# In[46]:


c


# # If else like statements
numpy.where(condition,x,y) choose element x or y depending on condition
# In[47]:


a = np.arange(10)


# In[48]:


a


# In[49]:


np.where(a<5,a,10*a)


# In[50]:


np.mean(a[a>5])


# # Ploating with Matplotlib 

# In[51]:


import matplotlib.pyplot as plt


# In[52]:


x = np.arange(0,10,0.2)
y = np.sin(x)


# In[58]:


x


# In[59]:


y


# In[54]:


fig , ax = plt.subplots()   #create a figure out an axes.
ax.plot(x,y, label='linear')


# In[56]:


plt.plot(x,y,label='linear')


# In[57]:


plt.close()


# In[61]:


x= np.arange(0,10,0.2)
y_sin=np.sin(x)
y_cos=np.cos(x)


# In[64]:


plt.plot(x, y_sin,label='Sine',linewidth=4.0)
plt.plot(x, y_cos, lable='Cos')

plt.xlabel('x label')
plt.ylabel('y label')
plt.title("Simple plot")
plt.legend()
plt.show()
plt.close()


# # Working with multiplot

# In[65]:


x = np.arange(0,10,0.2)


# In[66]:


y_sin =np.sin(x)
y_cos =np.cos(x)


# In[67]:


plt.subplot(2,1,1)
plt.plot(x,y_sin)
plt.title('Sine Plot')

plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('Cos Plot')


# In[68]:


plt.close()


# # Plotting images
imorting image data into Numpy arrays 
# In[1]:


import matplotlib.image as mpimg


# In[5]:


img=mpimg.imread('D:\image')


# In[6]:


imgplot = plt.imshow(img)


# # Applying pseudocolor schemes to image plots 
Pseudocolor can be a useful tool for enhancing contrast and visualizing your data more easily .its relevent single- channel, grayscale
# In[7]:


img.shape


# In[8]:


red_data=img[:,:,0]


# In[9]:


red_data.shape


# In[10]:


plt.imshow(red_data)


# In[11]:


plt.imshow(red_data,cmap="terrain")
plt.colorbar()   # color refrence
plt.close()


# # Enhancing Contrast
# 
Sometimes you want to enhance the contrast in your image, or expand the contrast in a particular region while sacrificing the detail in color that dont vary much , or dont matter.
# In[ ]:



plt.hist(red_data.ravel(),bins=256,range=(0.0,1.0))
plt.subplot(1,2,1)
plt.imgshow(red_data , clim=(0.1,0.3),cmap='terrain')
plt.subplot(1,2,2)
plt.imshow(red_data,cmap='terrain')

plt.close()

plt.figure(figure(10,10))
plt.subplot(1,3,1)
plt.imshow(img[:,:,0],cmap='terrain')

plt.close()

