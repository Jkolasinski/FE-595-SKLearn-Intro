#Jakub Kolasinski
#FE595
#SKLearn Assignment

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


#Part 1
def bostonCoeff():
    boston = datasets.load_boston()
    x = boston.data
    y = boston.target
    linreg = LinearRegression().fit(x,y)
    coeff_value  = max([abs(x) for x in linreg.coef_])
    variable = boston.feature_names[[abs(x) for x in linreg.coef_].index(coeff_value)]
    return "The feature with the largest coefficient is: ", variable, " with a value of: ", coeff_value


#Part 2
def irisKMeans():
    iris = datasets.load_iris()
    x = []
    y = []
    
    for i in range(1,10):
        k = KMeans(i).fit(iris.data,iris.target)
        x.append(i)
        y.append(k.inertia_)
    
    plt.plot(x,y)
    plt.xticks(x)
    plt.xlabel("Clusters")
    plt.ylabel("Inertia")
    plt.show()
    
    
if __name__=='__main__':
    print(bostonCoeff())
    irisKMeans()
    print("It appears that the dataset does represent 3 different varieties of iris, as the intertia greatly changes after 3 clusters. There does not appear to be much benefit in going beyond 3 clusters.")
    