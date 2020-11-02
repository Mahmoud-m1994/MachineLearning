import pandas as pd;
from pandas.plotting import scatter_matrix;
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# loading data from data.csv to pandas
mydata = pd.read_csv("/Users/mahmoudibrahim/IdeaProjects/machineLearning/Øving5/data");
#print(mydata)

# converting data to dummy
mydataDummy = pd.get_dummies(mydata)

# listing length of edibility_e based on habitat
habitat_d1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_d'] == 1])
habitat_g1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_g'] == 1])
habitat_l1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_l'] == 1])
habitat_m1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_m'] == 1])
habitat_p1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_p'] == 1])
habitat_u1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_u'] == 1])
habitat_w1 = len(mydataDummy[mydataDummy['edibility_e'] == 1] [mydataDummy['habitat_w'] == 1])

# listing length of edibility_p based on habitat
habitat_d0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_d'] == 1])
habitat_g0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_g'] == 1])
habitat_l0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_l'] == 1])
habitat_m0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_m'] == 1])
habitat_p0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_p'] == 1])
habitat_u0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_u'] == 1])
habitat_w0 = len(mydataDummy[mydataDummy['edibility_p'] == 1] [mydataDummy['habitat_w'] == 1])

# plotting figure of feature edibility_e
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

names1 = ['habitat_d', 'habitat_g', 'habitat_l','habitat_m','habitat_p','habitat_u','habitat_w']
values1 = [habitat_d1, habitat_g1, habitat_l1,habitat_m1,habitat_p1,habitat_u1,habitat_w1]
plt.figure(figsize=(16, 16))
index = np.arange(7)
plt.suptitle('edibility_e')
plt.bar(names1, values1)
plt.ylabel('edibility_e numbers')
plt.xticks(index, names1)
plt.show()

# plotting figure of feature edibility_p

values0 = [habitat_d0, habitat_g0, habitat_l0,habitat_m0,habitat_p0,habitat_u0,habitat_w0]
plt.figure(figsize=(16, 16))
index = np.arange(7)
plt.suptitle('edibility_e')
plt.bar(names1, values0)
plt.ylabel('edibility_e numbers')
plt.xticks(index, names1)
plt.show()

plt.spy(mydataDummy['population_v'==1], markersize=1)
fig = plt.gcf()
fig.set_size_inches(100,200)
plt.plot()
plt.show()