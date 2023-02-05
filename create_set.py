#implementation of data complexity estimation heuristics: http://people.sabanciuniv.edu/~berrin/cs512/reading/ho-complexity.pdf

import numpy as np
import h5py

def feature_mean_and_variance(data,labels,class_label):
	class_indices=np.where(labels==class_label)
	class_values=data[class_indices]
	return np.mean(class_values, axis=0), np.std(class_values, axis=0)

f = h5py.File('data/kagome_open_16.h5') #spin-ed-master/data/square_16_nonperiodic.h5

f_square = h5py.File('data/square_16_nonperiodic.h5') #spin-ed-master/data/square_16_nonperiodic.h5
#f_triangle
#f_kagome kagome_open_16.h5

a=f['basis']['representatives']
data=[list(np.binary_repr(x,16)) for x in a]
data_int=[[eval(i) for i in x] for x in data]
npdata=np.asarray(data_int)

labels=np.sign(f['hamiltonian']['eigenvectors'][0])

class_a=max(labels)
class_b=min(labels)

class_a_mean, class_a_std=feature_mean_and_variance(npdata,labels,class_a)
class_b_mean, class_b_std=feature_mean_and_variance(npdata,labels,class_b)

############ feature overlap measures #############
#Fisher's Discriminant Ratio 
fdr=[(class_a_mean[i]-class_b_mean[i])**2 /(class_a_std[i]**2+class_b_std[i]**2) for i in range(0,len(class_a_mean))]

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
within the classes to the gap between the classes. While
the MST-based measure is sensitive to which 5intra or inter
class) neighbor is closer to a point, this measure takes into
account the magnitudes of the differences.
'''




