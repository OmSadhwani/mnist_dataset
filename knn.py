from mnist import MNIST
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd





def main():
    
    data = MNIST('samples')

    X_train, y_train = data.load_training()
    X_test, y_test = data.load_testing()

    knn = KNeighborsClassifier(n_neighbors = 4)
    knn.fit(X_train, y_train)
    
    
    s= knn.score(X_test, y_test)
    print(s)



if __name__=='__main__':
    main()


