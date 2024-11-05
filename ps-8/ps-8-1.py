"""
Model the data set in survey.csv using the logistic function p(x) = (1 + exp(-(B0 + B1*x)))^(-1),
where x = the respondent's age and p(x) = the probability that they recognise the phrase "Be kind, rewind."
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize0

#Logistic function
def p(x, B0, B1):
    return np.power(1 + np.exp(-(B0 + B1 * x)), -1)

#Load in data from survey.csv
data = np.loadtxt('/content/survey.csv', delimiter=',', dtype=np.float64, skiprows=1)

ages = data[:,0]
classes = data[:,1]

#Get minimum and maximum ages for write-up purposes
# print(ages.min(), ages.max())
# >> 5.4 79.4

def cross_entropy(x):
    """
    Cross-entropy loss function.
    x: numpy.ndarray of length 2
        Should contain the parameters B0 and B1, as seen in the logistic function p(x, B0, B1) defined above.
    """
    B0 = x[0]
    B1 = x[1]
    return (-classes * np.log(p(ages, B0, B1)) - (1-classes) * np.log(1-p(ages, B0, B1))).sum()

#After plotting the data, I guess that the crossover age is about 42, meaning -B0/B1 = 42.
res = minimize(cross_entropy, [-4.5, .1])

print(f"B0: {res.x[0]}\t +/-\t{np.sqrt(res.hess_inv[0][0])}")
print(f"B1: {res.x[1]}\t +/-\t{np.sqrt(res.hess_inv[1][1])}\n")
print(f"Covariance matrix:\n{res.hess_inv}")

# >> B0: -5.620231349078274	 +/-	1.0504796296586572
# > B1: 0.10956336908552505	 +/-	0.020703727387864905

# >> Covariance matrix:
# >> [[ 1.10350745e+00 -2.08174241e-02]
# >>  [-2.08174241e-02  4.28644328e-04]]

#Plot model logistic along with real data
plt.scatter(ages, classes, label="Data")

y = np.linspace(ages.min(), ages.max(), 100)
plt.plot(y, p(y, res.x[0], res.x[1]), label="Model", color='darkorange')

plt.legend()
plt.title('Probability of Knowing the Phrase "Be kind, rewind" vs. Age')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.savefig("logistic.eps", format='eps')
plt.show()