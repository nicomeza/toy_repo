#!/usr/bin/env python
# coding: utf-8

# # Problem 1
# 
# Let's define :
# 
# $I = \int_{-\infty}^{\infty}{dx \exp(-x^2/2)}$
# 
# Then, 
# 
# $I^2 = \int_{-\infty}^{\infty}\int_{-\infty}^{\infty}{dxdy \exp(-(x^2+y^2)/2)}$
# 
# . We change to spherical coordinates,
# 
# $I^2 = \int_{0}^{2\pi}\int_{-\infty}^{\infty}{rdrd\theta \exp(-r^2/2)} = 2\pi \int_{-\infty}^{\infty}{rdr \exp(-r^2/2)} = 2\pi\int_{0}^{\infty}{du \exp(-u)} = 2\pi$.
# 
# Therefore, 
# 
# $I = \sqrt{2\pi}$
# 
# We can obtain the more general integral $I = \int_{-\infty}^{\infty}{dx \exp(-x^2/2\sigma^2)}$, calling $u = x/\sigma$, then $I = \sigma \int_{-\infty}^{\infty}{dx \exp(-x^2/2)}= \sqrt{2\pi\sigma^2}$.
# 
# Therefore, $N(a) = \sqrt{2\pi a}$
# 
# # Problem 2
# 
# Using Bayes theorem : 
# 
# $P(a|x) = P(x|a)P(a)/\int{da P(x|a)P(a)}$.
# 
# If we consider a flat prior from 0 to some large number L that goes to infinity we have: <br>
# $P(a|x) = (2\pi a)^{-1/2} \exp(-x^2/2a) / \int_{0}^{\infty}{da (2\pi a)^{-1/2} \exp(-x^2/2a)}$
# 
# We use the change of variable u = x^2/2a, which gives: <br>
# 
# $\frac{x}{2\sqrt{\pi}}\int_{0}^{\infty}{du u^{-3/2} \exp(-u)}$
# 
# We can write this using the extended gamma function for negative values: <br>
# 
# $\int_{0}^{\infty}{du u^{-3/2} \exp(-u)} = -\Gamma(-1/2) = -\Gamma(-1/2 + 1)/(-1/2) = 2\Gamma(1/2) = 2\sqrt{\pi}$
# 
# Finally we obtain the normalization as 
# 
# $\frac{x}{2\sqrt{\pi}}\int_{0}^{\infty}{du u^{-3/2} \exp(-u)} = x$
# 
# Now we write the posterior 
# 
# $P(a|x) = (2\pi a)^{-1/2} \exp(-x^2/2a) / x$
# 
# For a set of independent observations $\{x_i\}$, we can use the product property of the gaussian distribution and replace x with $\sqrt{\sum{x_i^2}}$
# 
# $P(\{x_i\}|a) = \prod_{i} (2\pi a)^{-1/2} \exp(-x_i^2/2a) = (2\pi a)^{-N/2} \exp(-\sum{x_i^2}/2a)$ 
# 
# Beyond the normalization factor the shape of the posterior for a flat prior in the whole range possible is **exactly** the same as the likelihood as a function of the parameter a.
# 
# # Problem 3

# In[2]:


import numpy as np
import numpy.random as random
import matplotlib.pyplot as pl

pi = np.pi


# In[31]:


N=2
sigma = 1
size=int(1e2)

x,y = random.multivariate_normal(np.zeros(N),np.diag(sigma**2. * np.ones(N)),size=size).T

pl.hist(x,alpha=0.5)
pl.hist(y,alpha=0.5)

H,xedges,yedges = np.histogram2d(x,y)

pl.figure()
pl.imshow(H.T,interpolation='nearest',origin='lower',extent=[xedges[0], xedges[-1],yedges[0],yedges[-1]])
pl.colorbar()


# In[70]:


N=1
sigma = 1
size=int(1e2)

X = random.multivariate_normal(np.zeros(N),np.diag(sigma**2. * np.ones(N)),size=size)

pl.hist(X,alpha=0.5)
np.std(X)


# # Problem 4

# In[83]:


def generator(a,sigma=0.05):
    
    return np.random.normal(a,sigma)

def loglikelihood(a,x):
    
    loglike = np.sum(np.log(1/np.sqrt(2*pi*a) * np.exp(-x**2./ (2*a))))
    return np.nan_to_num(loglike,nan=0.0)
    
def prior(a):
    
    return (np.array(a)>0).astype(int)

def posterior(a,x):
    
    return np.exp(loglikelihood(a,x))*prior(a)

def MCMC(seed,x,sigma=1):
    
    '''seed is the previous value in parameter space.
    x is the vector of observed values {x_i}.
    sigma is the meta-parameter controlling the width of the gaussian in the generator function.'''
    
    new = generator(seed,sigma)
    Pnew = posterior(new,x)
    Pold = posterior(seed,x)
    
    r = Pnew/Pold
    
    u = np.random.uniform(0,1)
    
    if r<u:
        
        return seed
    else:
        return new
    


seed = random.uniform(0.1,2)

steps = 100
sigma = 0.05


chain = []

for i in range(steps):
    
    sprout = MCMC(seed,X,sigma=sigma)
    chain.append(sprout)
    seed = sprout


# # Problem 5
# 
# Plotting the resulting chain.

# In[84]:


pl.plot(range(steps),chain)


# # Problem 6

# In[81]:


a_s = np.linspace(0.01,10,100)
# pl.plot(a_s,posterior(a_s,X))
pl.hist(chain,density=True)

np.mean(chain)


# In[ ]:




