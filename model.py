# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
This is a model of reaction norm evolution along a bet-hedging-plasticity
continuum. Other models on bet-hedging/plasticity either assume two reaction norms
evolving (1 to a predictable cue = plasticity, 1 to white noise; e.g. Tufto 2015),
or an independent contribution of 1 reaction norm and an instability locus 
(e.g. Scheiner2014). This model assumes that there is a single polyphenic reaction 
norm evolving, its shape determines whether we call this strategy bet-hedging
or plasticity (see also  https://doi.org/10.1101/752881 and 10.32942/osf.io/trg34).
The model simulates a multivoltine species that has to diapause before winter onset,
but winter onset is not predictable across years. Depending on the standard 
deviation of winter onset, more plasticity or more bet-hedging is expected to evolve.

A starting population has *popsize* individuals. Each individual has a different
genotype with four properties (*b,c,d,e*), whic determine logistic reaction norm shape 
in response to *t* (time, e.g. day length). In each time step t, each 
individual decides whether to go to diapause, based on its reaction norm shape. 
If diapausing the individual is transferred to a seed bank (egg bank) of infinite size; 
if not, it will remain and reproduce with a growth rate *growth_rate*. The offspring
inherit the same genotype, except for the possibilty of mutations with a rate of 
*mut_rate*. These mutations have quite large effects. 
When t reaches winter onset,  all individuals will be removed. The next year begins
and a new winter onset is drawn from a normal distribution with mean *mu_float* and 
standard deviation *sigma_float*. The seed bank replaces the population from last year. 
If the seed bank is larger than *popsize*, *popsize* individuals are randomly drawn 
from the seed bank. The model runs for *max_year* years.

The model uses variable notation as in the Python book by Punch and Enbody 
(similar to google notation), but "_int" is not appended to variables of class integer.
Class float is marked by "_float", "_rate", "_probability" or "_ratio" as appropriate.
'''


import random
import numpy
import math
import matplotlib.pyplot as plt

''' model constants'''
#added here empty to prevent errors; will be properly intialized under 'main program'
growth_rate = 0
popsize = 0
mut_rate=0
mu_float = 0
sigma_float = 0
max_year = 0

pop_list = []
year_count = 0
surv_list =[]
w_on_list = []


'''classes'''
class Individual (object):
    '''creates individual with 4 reaction norm properties and diapause switch.
    
    reaction norm is given by p(diap) = c + (d-c)/(1+math.exp(-b*(t-e))) with b =slope,
    c,d = lower and upper limit, e = inflection point'''
    
    def __init__(self,b,c,d,e): #all float
        self.b = b #-inf to +inf, though values >+-5 carry little meaning
        self.c = c #0 to 1
        self.d = d #0 to 1, though should generally be >c
        self.e = e #- inf to +inf, expected ~mu_float
        
    def __str__ (self):
        x = str("slope: {:.3f} lower limit: {:.3f} upper limit: {:.3f} midpoint: {:.3f}".format(
                self.b, self.c, self.d, self.e))
        return (x)
    
    def reproduce(self, r, mut_frac): 
        '''reproduces the individual r times, with slight deviations with p = mut_frac
        
        expected input: r = float > 0, mut_frac = float of range 0,1'''
        integ, fract = math.floor(r), r- math.floor(r)
        return_list = []
        if random.uniform(0,1) < fract:
            integ +=1 
            
            
        for i in range(1,integ): 
            #for r =3.5, the loop will run 2 times in 50% of cases, 3 times in the other 50%
            # = adult + 2.5 offspring
                       
            #add random mutation - mutations of large effects!
            new_b = self.b if random.uniform(0,1) > mut_frac else random.gauss(self.b,1)
            new_c = self.c if random.uniform(0,1) > mut_frac else random.uniform(0,1)
            new_d = self.d if random.uniform(0,1) > mut_frac else random.uniform(new_c,1)
            new_e = self.e if random.uniform(0,1) > mut_frac else random.gauss(self.e,1)
            return_list.append(Individual(new_b, new_c, new_d, new_e))
            #alternative c/d mutation: 
            #new_c = self.c if random.uniform(0,1) > mut_frac else random.gauss(self.c,0.1)
            #new_d = self.d if random.uniform(0,1) > mut_frac else random.gauss(self.d,0.1)
            #if new_c < 0:
            #    new_c = 0
            #if new_c > 1:
            #    new_c = 1
            #if new_d < new_c: #prevents upper bound from being smaller than lower bound
            #    new_d = new_c
            #if new_d > 1:
            #    new_d = 1     
        return (return_list)
    
    def check_diap (self, t_int):
        '''test for diapause given individual's reaction norm shape and t'''
        diap_probability = self.c + (self.d-self.c)/(1+math.exp(-self.b*(t_int-self.e))) 
        diap = bool(numpy.random.binomial(1,diap_probability))
        return(diap)
        
    def var_among(self):
        ''' calculate variance among environments (plasticity)'''
        percs =[]
        for t in range(50):
            upper = (self.d-self.c) #=upper part of diap_probability equation
            lower = 1 + math.exp(-1*self.b* (t-self.e)) #lower part
            percs.append(round(self.c + (upper/lower), 4))#note constant c
            #rounding included for numerical stability; sd(10^-130,10^-15, 0.2) etc
        return(numpy.std(percs,ddof=1))
        
    def var_within(self):
        percs= []
        for t in range(50):
            upper = (self.d-self.c)
            lower = 1 + math.exp(-1*self.b* (t-self.e)) 
            val = round(self.c + (upper/lower), 4)
            percs.append(round(val *  (1-val),4))
        return(numpy.mean(val))
        
    
          
    
class Year(object):
    '''decisions of diapause and reproduction for 1 year
    
    input: mu = mean winter onset, sigma = sd winter onset, both float; eggs = list,
    each entry one instance of class Individual'''
    def __init__ (self, mu, sigma, popsize, eggs):
        self.t_on = random.normalvariate(mu, sigma)
        self.t_on = round(self.t_on)
        self.awake_list = eggs if len(eggs) < popsize else random.sample(eggs, popsize)
        self.diapause_list = []
    
    def __str__(self):
        return ("Winter onset on day {}\nIn diapause:{}; awake:{}".format(
                self.t_on, len(self.diapause_list),len(self.awake_list)))
    
    def runday(self, gr_float, mut_float, t):
        '''on each day, every Individual may enter diapause; if awake, it reproduces
        
        input: growth rate, mutation rate, both float; and t(int)'''
        offspring_list= []
        remove_from_awake_list =[]
        for individual in self.awake_list:
            if individual.check_diap(t): #line both checks for diapause and returns outcome
                self.diapause_list.append(individual)
                remove_from_awake_list.append(individual)
           #     print("diapause!", individual)
            else:
                offspring_list.extend(individual.reproduce(gr_float, mut_float))
                #extend instead of append here because return of reproduce is a list, not
                #an individual instance of Individual
        #print(len(offspring_list),"new offspring!")
        for rem in remove_from_awake_list:
            self.awake_list.remove(rem)
        self.awake_list.extend(offspring_list)
    
    def runyear(self, growth_rate, mut_rate):
        '''do daily stuff until winter arrives; then kill all non-diapausing Individuals'''
        
        for t in range(self.t_on):
            self.runday(growth_rate, mut_rate, t)
           # print ("day ", t, "done. awake:", len(self.awake_list), "diapause:", len(self.diapause_list))
      #  print ("winter arrived at day: ", t)
        w_on_list.append(t)
        surv_list.append(len(self.diapause_list))
    

'''main program'''
growth_rate = 1.2
popsize = 200
mut_rate = 1/popsize #mutation rate
mu_float = 20
sigma_float = 0
max_year = 1000


year_count = 0
surv_list =[]
w_on_list = []
res_list = []

def run_program(growth_rate = growth_rate, popsize = popsize, mut_rate = mut_rate,
                mu_float= mu_float, sigma_float = sigma_float, max_year = max_year):
    #initialize:
    pop_list =[]
    for i in range(popsize):
        pop_list.append(Individual(0.5,0,1,20))
    init = Year(mu_float, sigma_float, popsize, pop_list)
    init.diapause_list = init.awake_list
    y_list = []
    y_list.append(init)
    print ("running.")
    for i in range(1, max_year):
        #print ("year ", i)
        y_list.append(Year(mu_float, sigma_float, popsize, y_list[i-1].diapause_list))
        #diapausing eggs of last year become new year's population (and will be 
        #reduced to popsize at year initialization))
        y_list[i].runyear(growth_rate, mut_rate)
        print(".", end = '')
    return(y_list)
        
        
def get_summary(output):
    '''provide population-level summary of reaction norm shapes'''
    #reaction_norms = [[],[],[],[],[],[]]
    winter = []
    for year in output:
        winter.append([year.t_on, len(year.diapause_list), len(year.awake_list)])
    return(winter)

def get_results(output):
    reaction_norm = []
    for year in output:  
        print("-",end="")
        b_list = []
        c_list = []
        d_list =[]
        e_list =[]
        within_list = []
        among_list =[]
        
        for individual in year.diapause_list:
            b_list.append(individual.b)
            c_list.append(individual.c)
            d_list.append(individual.d)
            e_list.append(individual.e)
            within_list.append(individual.var_within())
            among_list.append(individual.var_among())
            
        reaction_norm.append([numpy.mean(b_list), numpy.mean(c_list), numpy.mean(d_list),
                             numpy.mean(e_list), numpy.mean(within_list), numpy.mean(among_list)])
    
    return(reaction_norm)
            
        
def plot_stuff():
    pass

y_list = run_program()
print("program done.\n\n")
w = get_summary(y_list)
res = get_results(y_list)


x_list = [i for i in range(len(w))]
w_on = [i[0] for i in w]
surv = [i[1]/1000 for i in w]
dead = [i[2]/1000 for i in w]

#res = get_results(y_list)
among =  [i[5] for i in res]
b =  [i[0] for i in res]
c =  [i[1] for i in res]
d =  [i[2] for i in res]
e =  [i[3] for i in res]


plt.plot(x_list, among, label = "variance among" )
plt.plot(x_list, b, label = "b" )
plt.plot(x_list, c, label = "c" )
plt.plot(x_list, d, label = "d" )
plt.plot(x_list, e, label = "e" )
plt.legend()
plt.show()'''
