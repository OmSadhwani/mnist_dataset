from mnist import MNIST
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd





def main():
    
    data = MNIST('samples')
    X_test=pd.read_csv('test.csv')
    
    X_train, y_train = data.load_training()
    #X_test, y_test = data.load_testing()

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train, y_train)
    
    
    predictions=knn.predict(X_test)
    predictions=pd.DataFrame(predictions)
    predictions.index=np.arange(1,28001)
    predictions.column=['Label']
    predictions.index.name='ImageId'
    predictions.to_csv('mnist1.csv',index=True)
    



    # s= knn.score(X_test, y_test)
    # print(s)



if __name__=='__main__':
    main()


