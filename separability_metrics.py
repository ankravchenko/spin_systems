#implementation of data complexity estimation heuristics: http://people.sabanciuniv.edu/~berrin/cs512/reading/ho-complexity.pdf
#look at table 1 at p5
#topological https://epub.ub.uni-muenchen.de/93712/1/MA_Gauss_komp.pdf
#https://arxiv.org/pdf/1808.03591.pdf

import numpy as np
import h5py
'''
from sklearn.preprocessing import StandardScaler

import itertools


import umap.umap_ as umap
reducer = umap.UMAP()
'''
def feature_mean_and_variance(data,labels,class_label):
	class_indices=np.where(labels==class_label)
	class_values=data[class_indices]
	return np.mean(class_values, axis=0), np.std(class_values, axis=0)

def find_nearest_distance(data, point):
	dist=[]
	for point_i in data:
		edist = np.linalg.norm(point-point_i)
		if edist!=0:
			dist.append(edist)
	return min(dist)		
	#calculate Eucleadean distance to all other points
	
	#pick min (exclude comparison to self)

def linear_combination(data):
	l=len(data)
	lc=[]
	for i in range (0,l):
		c1=random.uniform(0, 1)
		c2=random.uniform(0, 1)
		i1=random(0,l)
		i2=random(0,l)
		lc[i] = (data[i1]*c1 + data[i2]*C1)/2


#f = h5py.File('data/square_16_nonperiodic.h5') #spin-ed-master/data/square_16_nonperiodic.h5
print('square:')
f = h5py.File('data/square_16_nonperiodic.h5') #spin-ed-master/data/square_16_nonperiodic.h5
f_kagome = h5py.File('data/kagome_open_16.h5') #spin-ed-master/data/square_16_nonperiodic.h5

#f_triangle
#f_kagome kagome_open_16.h5

a=f['basis']['representatives']
data=[list(np.binary_repr(x,16)) for x in a]
data_int=[[eval(i) for i in x] for x in data]
npdata=np.asarray(data_int)


labels=np.sign(f['hamiltonian']['eigenvectors'][0])

transform=np.random.randint(1, 5, size=(len(npdata[0])*10,len(npdata[0])))

npdata=np.asarray([transform.dot(x) for x in npdata])

idx = np.random.choice(np.arange(len(labels)), 1000, replace=False)
data_sample = npdata[idx]
labels_sample = labels[idx]


class_a=max(labels)
class_b=min(labels)

class_a_indices=np.where(labels==class_a)
class_a_values=npdata[class_a_indices]
class_b_indices=np.where(labels==class_b)
class_b_values=npdata[class_b_indices]


class_a_indices_sample=np.where(labels_sample==class_a)
class_a_values_sample=data_sample[class_a_indices_sample]
class_b_indices_sample=np.where(labels_sample==class_b)
class_b_values_sample=data_sample[class_b_indices_sample]


class_a_mean, class_a_std=feature_mean_and_variance(npdata,labels,class_a)
class_b_mean, class_b_std=feature_mean_and_variance(npdata,labels,class_b)

############ feature overlap measures #############
#Fisher's Discriminant Ratio 
fdr=[(class_a_mean[i]-class_b_mean[i])**2 /(class_a_std[i]**2+class_b_std[i]**2) for i in range(0,len(class_a_mean))]

#print('FDR: ', fdr)
#volume of overlap regions - not applicable, value ranges are the same

#feature efficiency: to do. FIXME

############ measures of separability of classes #############

#UMAP of all 3 sets! FIXME

#linear separability (duh)

#mixture identifiability

'''A closely related measure is defined as follows: We first
compute the Euclidean distance from each point to its
nearest neighbor within or outside the class. We then take
the average of all the distances to intraclass nearest
neighbors, and the average of all the distances to interclass
nearest neighbors. The ratio of the two averages is used as a
measure 5N2). This measure compares the dispersion
within the classes to the gap between the classes. 
'''

interclass=[]
intraclass=[]

for i in range(0,len(labels_sample)):
		#if (i%500 == 0):
		#	print(i, ' processed') 
		a_distance=find_nearest_distance(class_a_values_sample, data_sample[i])
		b_distance=find_nearest_distance(class_b_values_sample, data_sample[i])
		mindist=min(a_distance,b_distance)
		i_class=labels[i]
		if labels[i] == class_a:
			if a_distance<b_distance:
				interclass.append(mindist)
			else:
				intraclass.append(mindist)
		if labels[i] == class_b:
			if a_distance<b_distance:
				intraclass.append(mindist)
			else:
				interclass.append(mindist)		

interclass_m=np.mean(interclass)
intraclass_m=np.mean(interclass)
m_ident=interclass_m/intraclass_m
print('mixture identifiability = ', m_ident)
#######Measures of Geometry, Topology, and Density of Manifolds##########

#Nonlinearity



#Space Covering by epsilon-Neighborhoods 

'''
The relevance of other measures is less obvious. For
example, it is not clear what can be inferred from the
intrinsic dimensionality of a problem without differentia-
tion by class. A problem can be very complex even if
embedded in a low-dimensional space 5e.g., randomly
labeled points along a one-dimensional space have a
complex class boundary). Also, variation in density within
a manifold seems irrelevant as long as the manifolds can be
easily separated. Similarly, existences of submanifolds of
one class surrounding those of the other 5e.g., consider two
classes black and white on a checkerboard) may make a
problem difficult for, say, a linear classifier, but may not
affect a nearest-neighbor classifier by much. Nevertheless,
in discussions on curse of dimensionality, the number of
samples is often compared to the number of feature
dimensions. To relate to such discussions, we include the
average number of samples per dimension as another
measure 5T2).
'''
