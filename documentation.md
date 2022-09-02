# Task
The task is to  apply supervised machine learning methods to the MNIST dataset and achieve at least 90% test accuracy
     
The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

###  K-Nearest Neighbors Classifier (KNN)
kNN is an algorithm which finds a group of k objects in the training set that are closest to the test object and bases the assignment of a label on the predominance of a class in this neighborhood.

To classify an unlabeled image 
* the distance of this image to the labeled objects (training data) is computed
* its k-nearest neighbors are identified and sorted in descending order according to
similarities computed
* the most occurred nearest neighbor in top k neighbors is then assigned as the
class label of the object

#### Accuracy vs k 
| k     | Accuracy |
|:------|:---------|
|1      |0.9691    |
|2      |0.9627    |
|3      |0.9705    |
|4      |0.9682    |
|5      |0.9688    |
|7      |0.9694    | 
|10     |0.9665    |
|15     |0.9633    |
|20     |0.9625    |
|50     |0.9534    |
|75     |0.9474    |
|100    |0.944     |


As we can see that as we increase the value of k above 10 the accuracy starts decreasing. This is due to the problem of overfitting
So we need to find the optimal value of k by running the code for different values of k.


There are some disadvantages of knn 
* It is computationally expensive because the algorithm compares the test data with all examples in training data and then finalizes the label
* The value of K is unknown and can be found by hit and trial methods
* High memory requirement – because all the training data is stored

