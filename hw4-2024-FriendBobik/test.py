from mixfit import max_likelihood, em_double_gauss

import unittest
import numpy as np
from scipy.stats import uniform
import matplotlib.pyplot as plt

tau = .25
mu1 = .5
sigma1 = .2
mu2 = 3.5
sigma2 = .7
n = 1e6

x1 = np.random.normal(mu1, sigma1, int(tau*n))
x2 = np.random.normal(mu2, sigma2, int((1-tau)*n))
x = np.r_[x1, x2]

#plt.hist(x)
#plt.show()
    

class MixFitTest(unittest.TestCase):
    def test_likelihood(self):
        new_tau, new_mu1, new_sigma1, new_mu2, new_sigma2 = max_likelihood(
            x, 0.5, np.mean(x)-np.std(x), np.std(x), 
            np.mean(x)+np.std(x), np.std(x))
        places = 2
        self.assertAlmostEqual(tau, new_tau, places=places)
        self.assertAlmostEqual(mu1, new_mu1, places=places)
        self.assertAlmostEqual(sigma1, new_sigma1, places=places)
        self.assertAlmostEqual(mu2, new_mu2, places=places)
        self.assertAlmostEqual(sigma2, new_sigma2, places=places)

    def test_em_double_gauss(self):
        new_tau, new_mu1, new_sigma1, new_mu2, new_sigma2 = em_double_gauss(
            x, 0.5, np.mean(x)-np.std(x), np.std(x), 
            np.mean(x)+np.std(x), np.std(x))
        places = 2

        if tau < 0.5:
            self.assertAlmostEqual(tau, new_tau, places=places)
            self.assertAlmostEqual(mu1, new_mu1, places=places)
            self.assertAlmostEqual(sigma1, new_sigma1, places=places)
            self.assertAlmostEqual(mu2, new_mu2, places=places)
            self.assertAlmostEqual(sigma2, new_sigma2, places=places)
        else: 
            self.assertAlmostEqual(tau, 1-new_tau, places=places)
            self.assertAlmostEqual(mu1, new_mu2, places=places)
            self.assertAlmostEqual(sigma1, new_sigma2, places=places)
            self.assertAlmostEqual(mu2, new_mu1, places=places)
            self.assertAlmostEqual(sigma2, new_sigma1, places=places)
    

if __name__ == "__main__":
    unittest.main()
