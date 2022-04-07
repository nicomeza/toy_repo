#!/usr/bin/env python
# coding: utf-8

# In[63]:


from matplotlib import pyplot as pl
import random
import numpy as np


# ## Define the neighborhoods for our 2 neighbor scenario with 3 possible inputs.
# 

# In[64]:


neighborhoods = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
set_length = len(neighborhoods)


# ## Now pick up a rule and write it as binary. In this case we choose a rule up to 3^9

# In[54]:



def to_ternary(n,length=False):
    
    '''Convert number in base 10 base 3. Returns a string that represents the ternary number. 
    If length is an integer the representation will have a number of digits given by length. Input must be positive.'''

    if length: 
        
        if n <= 0:
            return '0'*length   
        else:
            s = {str(i):0 for i in range(length)}
            
    else:
        
        if n <= 0:
            return '0'    
        else:
            i = int(np.log(n)/np.log(3))
            s = {str(i):0 for i in range(i+1)}

    while n>0:
    
        i = int(np.log(n)/np.log(3))

        if str(i) in s.keys():
            s[str(i)] += 1 
        else:
            s[str(i)] = 1 
               
        n -= 3**i
    
    return ''.join([str(x) for x in s.values()])[::-1]


to_ternary(0,length=9)  
to_ternary(1,length=9)  
to_ternary(1331,length=9)  


# ## Define a rule 

# In[135]:


rule = 1331

in_ternary = to_ternary(rule,length=set_length)

print (in_ternary)
# create the lookup table dictionary
lookup_table = {}
for i in range(set_length):
    key = neighborhoods[i]
    val = in_ternary[i]
    lookup_table.update({key:val})
    
lookup_table


# # Set initial conditions and iterate

# In[136]:


length = 100
time = 40

initial_condition = [random.randint(0,2) for _ in range(length)]
print('Initial config : %s'%initial_condition)


# initialize spacetime field and current configuration
spacetime_field = [initial_condition]
current_configuration = initial_condition.copy()

for t in range(time):
    
    new_configuration = []
    
    for i in range(length):
        
        neighborhood = (current_configuration[(i-1)%length], 
                        current_configuration[i])
        
        new_configuration.append(int(lookup_table[neighborhood]))
        
    current_configuration = new_configuration
    spacetime_field.append(new_configuration)
        
    


# In[137]:


spacetime_field

pl.figure(figsize=(12,12))
pl.imshow(spacetime_field, cmap=pl.cm.Greys, interpolation='nearest')
pl.show()


# # Now let's try to implement classes

# In[127]:


def spacetime_diagram(spacetime_field, size=12, colors=pl.cm.Greys):
    '''
    Produces a simple spacetime diagram image using matplotlib imshow with 'nearest' interpolation.
    
   Parameters
    ---------
    spacetime_field: array-like (2D)
        1+1 dimensional spacetime field, given as a 2D array or list of lists. Time should be dimension 0;
        so that spacetime_field[t] is the spatial configuration at time t. 
        
    size: int, optional (default=12)
        Sets the size of the figure: figsize=(size,size)
    colors: matplotlib colormap, optional (default=plt.cm.Greys)
        See https://matplotlib.org/tutorials/colors/colormaps.html for colormap choices.
        A colormap 'cmap' is called as: colors=plt.cm.cmap
    '''
    pl.figure(figsize=(size,size))
    pl.imshow(spacetime_field, cmap=colors, interpolation='nearest')
    pl.show()

def unit_seed(margin_length):
    '''
    Returns a list of a single '1' or '2' bounded by margin_length number of '0's 
    on either side. 
    
    Parameters
    ----------
    margin_length: int
        Number of zeros bounding the central one on either side. 
        
    Returns
    -------
    out: list
        [0,]*margin_length + [1,] + [0,]*margin_length
    '''
    if not isinstance(margin_length, int) or margin_length < 0:
        raise ValueError("margin_length must be a postive int")
    
    
    return [0,]* margin_length + [random.randint(1,2),] + [0,]*margin_length



def lookup_table(rule_number):
    '''
    Returns a dictionary which maps ECA neighborhoods to output values. 
    Uses Wolfram rule number convention.
    
    Parameters
    ----------
    rule_number: int
        Integer value between 0 and 255, inclusive. Specifies the ECA lookup table
        according to the Wolfram numbering scheme.
        
    Returns
    -------
    lookup_table: dict
        Lookup table dictionary that maps neighborhood tuples to their output according to the 
        ECA local evolution rule (i.e. the lookup table), as specified by the rule number. 
    '''
    if not isinstance(rule_number, int) or rule_number < 0 or rule_number >= 3**9:
        raise ValueError("rule_number must be an int between 0 and 3^9-1, inclusive")
    neighborhoods = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    in_ternary = to_ternary(rule_number,length=9)
    
    return dict(zip(neighborhoods, map(int,in_ternary))) # use map so that outputs are ints, not strings


class ECA(object):
    '''
    Elementary cellular automata simulator.
    '''
    def __init__(self, rule_number, initial_condition):
        '''
        Initializes the simulator for the given rule number and initial condition.
        
        Parameters
        ----------
        rule_number: int
            Integer value between 0 and 255, inclusive. Specifies the ECA lookup table
            according to the Wolfram numbering scheme.
        initial_condition: list
            Binary string used as the initial condition for the ECA. Elements of the list
            should be ints. 
        
        Attributes
        ----------
        lookup_table: dict
            Lookup table for the ECA given as a dictionary, with neighborhood tuple keys. 
        initial: array_like
            Copy of the initial conditions used to instantiate the simulator
        spacetime: array_like
            2D array (list of lists) of the spacetime field created by the simulator.
        current_configuration: array_like
            List of the spatial configuration of the ECA at the current time
        '''
        # we will see a cleaner and more efficient way to do the following when we introduce numpy
        for i in initial_condition:
            if i not in [0,1,2]:
                raise ValueError("initial condition must be a list of 0s,1s or 2s")
                
        self.lookup_table = lookup_table(rule_number)
        self.initial = initial_condition
        self.spacetime = [initial_condition]
        self.current_configuration = initial_condition.copy()
        self._length = len(initial_condition)

    def evolve(self, time_steps):
        '''
        Evolves the current configuration of the ECA for the given number of time steps.
        
        Parameters
        ----------
        time_steps: int
            Positive integer specifying the number of time steps for evolving the ECA.  
        '''
        if time_steps < 0:
            raise ValueError("time_steps must be a non-negative integer")
        # try converting time_steps to int and raise a custom error if this can't be done
        try:
            time_steps = int(time_steps)
        except ValueError:
            raise ValueError("time_steps must be a non-negative integer")

        for _ in range(time_steps): # use underscore if the index will not be used
            new_configuration = []
            for i in range(self._length):

                neighborhood = (self.current_configuration[(i-1)], 
                                self.current_configuration[i])

                new_configuration.append(self.lookup_table[neighborhood])

            self.current_configuration = new_configuration
            self.spacetime.append(new_configuration)
            
    def print_map(self):
        
        print(self.lookup_table)


# In[133]:


rule_ = ECA(11111, unit_seed(50))
rule_.print_map()
rule_.evolve(100)
spacetime_diagram(rule_.spacetime, 10)


# In[ ]:




