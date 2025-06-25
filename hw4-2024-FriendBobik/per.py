from astropy.coordinates import SkyCoord
from astroquery.utils.tap import TapPlus
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import numpy as np
import json


from mixfit import em_double_cluster, em_double_2d_gauss






limit_magnitude = 16
center_coord = SkyCoord('02h21m00s +57d07m42s')
target_size = 0.5
range_ra = (center_coord.ra.degree - target_size / np.cos(center_coord.dec.radian),
            center_coord.ra.degree + target_size / np.cos(center_coord.dec.radian))
range_dec = (center_coord.dec.degree - target_size, center_coord.dec.degree + target_size)
query = '''
SELECT ra, dec, pmra, pmdec FROM gaiadr3.gaia_source WHERE phot_bp_mean_mag < {:.2f} AND pmra IS NOT NULL AND pmdec IS NOT NULL AND ra BETWEEN {:} AND {:} AND dec BETWEEN {:} AND {:}
'''.format(limit_magnitude, *range_ra, *range_dec)

gaia = TapPlus(url="https://gaia.aip.de/tap", verbose=True)
job = gaia.launch_job_async(query)
stars = job.get_results()
                                    # Этих коментраиев нет ОЧЕНЬ не хватало, было сложно понять что происходит
ra    = np.asarray(stars['ra']._data)   # Массив прямого восхождения
dec   = np.asarray(stars['dec']._data) # Массив склонений
pmra  = np.asarray(stars['pmra']._data) # Массив собственного движения в RA
pmdec = np.asarray(stars['pmdec']._data) # Массив собственного движения в Dec




x = np.vstack([pmra, pmdec]).T
tau_init    = np.random.rand() + 0.01
mu1_init    = np.mean (x, axis=0)
sigma1_init = np.std  (x, axis=0)
mu2_init    = np.mean (x, axis=0)
sigma2_init = np.std  (x, axis=0)




tau, mu1, sigma1, mu2, sigma2 = em_double_2d_gauss(x, tau_init, mu1_init, sigma1_init, mu2_init, sigma2_init)

data = {
    "ratio": round(tau.tolist(),2),
    "motion": {
        "cluster": {"ra": round(mu1.tolist()[0],2), "dec": round(mu1.tolist()[1],2)},
        "background": {"ra": round(mu2.tolist()[0],2), "dec": round(mu2.tolist()[1],2)}
    }
}

with open('per.json', 'w') as f:
    json.dump(data, f, indent=2)





p1 = tau * multivariate_normal.pdf(x, mu1, np.diag([sigma1[0]**2, sigma1[0]**2]))
p2 = (1 - tau) * multivariate_normal.pdf(x, mu2, np.diag([sigma2[0]**2, sigma2[0]**2]))
gamma = p1 / (p1 + p2)




plt.figure(figsize=(10, 8))

plt.scatter(ra, dec, c=gamma)
plt.xlabel('Прямое восхождение')
plt.ylabel('Склоненение')
plt.title('Рассеяния точек звёздного поля')

plt.tight_layout()
plt.savefig('per1.png')


plt.figure(figsize=(10, 8))

plt.scatter(pmra, pmdec, c=gamma)
plt.xlabel('Движение по прямому восхождению')
plt.ylabel('Движение по склонению')
plt.title('Рассеяния собственных движений')
circle1 = Circle((mu1[0], mu1[1]), sigma1[0], fill=False) 
circle2 = Circle((mu2[0], mu2[1]), sigma2[0], fill=False)

plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.tight_layout()
plt.savefig('per2.png')

# с цветами какая-то фигня ¯\_(ツ)_/¯