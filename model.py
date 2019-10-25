# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
This is a model of reaction norm evolution along a bet-hedging-plasticity
continuum. The model simulates a multivoltine species that has to diapause before winter onset,
but winter onset is not predictable across years. On the one hand, continued reproduction
leads to exponential growth; on the other hand, a too late diapause decision kills the genotype.
The decision to diapause is modelled as logistic reaction norm (probability of diapause ~ day length),
and depending on the standard deviation of winter onset, more plasticity or more bet-hedging is 
expected to evolve.
If winter onset is predictable (standard deviation = 0), the reaction norm is expected
to evolve a steep slope (plasticity), inducing diapause just before mean winter onset. If the 
standard deviation is high (winter unpredictable), the reaction norm may evolve an early inflection
point (conservative bet-hedging), or a flat slope / low range (e.g. >10% diapause probability
under any day length), such that some offspring will be diapausing regardless of winter onset
(diversified bet-hedging). The shape of the reaction norm can be summarised by variance among 
environments and variance within environments (see also  https://doi.org/10.1101/752881 and 
10.32942/osf.io/trg34).

A starting population has *popsize* individuals. Each individual has a different
genotype with four properties (*b,c,d,e*), whic determine logistic reaction norm shape 
in response to *t* (time, e.g. day length). In each time step t, each 
individual decides whether to go to diapause, based on its reaction norm shape. 
If diapausing the individual is transferred to a seed bank (egg bank) of infinite size; 
if not, it will remain and reproduce with a growth rate *growth_rate*. The offspring
inherit the same genotype, except for the possibilty of mutations with a rate of 
*mut_rate*.
When t reaches winter onset,  all individuals will be removed. The next year begins
and a new winter onset is drawn from a normal distribution with mean *mu_float* and 
standard deviation *sigma_float*. The seed bank replaces the population from last year. 
If the seed bank is larger than *popsize*, *popsize* individuals are randomly drawn 
from the seed bank. The model runs for *max_year* years.
The reaction norm is expected to evolve in response to *sigma_float* and *mu_float*.
Highly plastic reaction norms have a steep slope (b), low lower limit(c) and high 
upper limit(d), and the midpoint(e) lies close to *mu_float*. Diversified bet-hedging 
reaction norms typically have a low b, or high c; a low d would be risk-seeking
strategy

The model uses variable notation as in the Python book by Punch and Enbody 
(similar to google notation), but "_int" is not appended to variables of class integer.
Class float is marked by "_float", "_rate", "_probability" or "_ratio" as appropriate.
'''


import random
import numpy
import math
import matplotlib.pyplot as plt

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
        x = f"slope: {self.b:.3f} lower limit: {self.c:.3f} upper limit: {self.d:.3f} midpoint: {self.e:.3f}"
        return (x)
    
    def reproduce(self, r, mut_frac): 
        '''reproduces the individual r times, with slight deviations with p = mut_frac
        
        expected input: r = float > 0, mut_frac = float of range 0,1'''
        integ, fract = math.floor(r), r- math.floor(r) #integer part and fractional part
        return_list = []
        if random.uniform(0,1) < fract:
            integ +=1 
            
            
        for i in range(1,integ): 
            #for r =3.5, the loop will run 2 times in 50% of cases, 3 times in the other 50%
            # = adult + 2.5 offspring
                       
            #add random mutation
            new_b = self.b if random.uniform(0,1) > mut_frac else \
            min(random.gauss(self.b,0.1),10) #min makes sure value stays below 10
            #b must be < 10, because very steep slopes cause math.range error in .var_comps()
            #math.exp(b*e) can become too large to handle
            new_c = self.c if random.uniform(0,1) > mut_frac else \
            min(max(0,random.gauss(self.c,0.1)),1) # 0< c < 1
            new_d = self.d if random.uniform(0,1) > mut_frac else \
            min(max(new_c,random.gauss(self.d,0.1)),1)     #c<d<1
            new_e = self.e if random.uniform(0,1) > mut_frac else \
            random.gauss(self.e,0.1)
            return_list.append(Individual(new_b, new_c, new_d, new_e))
        return (return_list)
    
    def check_diap (self, t_int):
        '''test for diapause given individual's reaction norm shape and t'''
        diap_probability = self.c + (self.d-self.c)/(1+math.exp(-self.b*(t_int-self.e))) 
        diap = bool(numpy.random.binomial(1,diap_probability))
        return(diap)
        
    def var_comps(self):
        ''' calculate variance among and within environments
        
        var_within = sum p*(1-p) / n
        var_among = sd^2 / n'''
        
        probability = []
        p2 = []
        for t in range(50):
            upper = (self.d-self.c) #=upper part of diap_probability equation
            lower = 1 + math.exp(-1*self.b* (t-self.e)) #lower part
            prob = round(self.c + (upper/lower), 4) #rounding for numerical stbility
            probability.append(prob)
            p2.append(prob * (1-prob))
            
        self.among = numpy.std(probability, ddof=1)
        self.within = numpy.mean(p2)    
          
    
        #only self.diapause_list remains important            
    

class Run_Program(object):
    '''run the model'''
    def __init__ (self, growth_rate = 1.2, popsize = 250, mut_rate = 1/250,
                mu_float = 20, sigma_float = 1, max_year = 2000, model_name = "generic"):
        '''saves parameters and creates starting population'''
        self.growth_rate = growth_rate
        self.popsize = popsize
        self.mut_rate = mut_rate
        self.mu_float = mu_float
        self.sigma_float = sigma_float
        self.max_year = max_year
        self.model_name = model_name
        
        self.t_on_list = [] #stores winter onsets of each year
        self.surv_list = [] #stores no. survivors of each year
        self.result_list =[]#stores mean reaction norm properties of each year
    
    def initialize_pop(self):
        pop_list = []
        for i in range(self.popsize):
            b = random.gauss(1,0.5)
            c = random.uniform(0,0.5)
            d= random.uniform(0.5,1)
            e = random.gauss(self.mu_float,self.sigma_float * 2)
            pop_list.append(Individual(b,c,d,e))
        return(pop_list)
        
    def __str__(self):
        first_line = "PARAMETERS\n"
        second_line = str(" r: {:5.2f}      N:{:4}    mutrate: {:4.2f} \n".format(
                self.growth_rate, self.popsize,self.mut_rate))
        third_line = str("mu: {:4.2f}  sigma: {:4.2f}   end: {:8}".format(
                self.mu_float, self.sigma_float, self.max_year))
        return(first_line+second_line+third_line)
        
        
    def run(self):
        print ("Running. 1 dot = 10 years")
        yc = 1
        eggs = self.initialize_pop()    #stores Year instances
        
        for i in range(1, self.max_year): 
            t_on = random.normalvariate(self.mu, self.sigma)
            t_on = round(t_on)
            self.t_on_list.append(t_on)
            
            awake_list = eggs if len(eggs) < self.popsize else\
            random.sample(eggs, self.popsize)
            
            if not awake_list:
                print ("population extinct!")
                yc+=1
            else:
                eggs = self.runyear(awake_list, t_on)          
                self.result_list.append(self.save_all_results())
                self.surv_list.append(len(eggs))
                yc +=1
                if not yc % 10: #if yc % 10 == 0
                    print(".", end = '')
        return(yc)
        
    def runyear(self, spring_list, end):
        '''on each day, every Individual may enter diapause; if awake, it reproduces
        
        input: growth rate, mutation rate, both float; and t(int)'''
        diapause_list =[]
        curr_list= spring_list
        for t in range(end):
            offspring_list = []
            for individual in curr_list:
                if individual.check_diap(t):
                    individual.var_comps() #time-intensive, so only performed when needed
                    #(implicitly skips those that don't make it to diapause before winter onset)
                    diapause_list.append(individual)
                else:
                    offspring_list.extend(individual.reproduce(self.growth_rate, self.mut_rate))
                #extend instead of append here because return of reproduce is a list, not
                #an individual instance of Individual
            curr_list = offspring_list
        return (diapause_list)
        

    def save_result(self, y, output = "among"):
        '''provide population level summary of reaction norm shape
    
        input: "b", "c", "d", "e", "among" or "within". any other input throws error'''
        ind_list =[]
        for individual in y.diapause_list:  
                ind_list.append(individual.__dict__[output]) #individual.b or individual.among...
        return(numpy.mean(ind_list))
    
    def save_all_results(self,y):
        '''not implemented'''
        b_list =[]        
        c_list =[]     
        d_list =[]     
        e_list =[]     
        among_list =[]     
        within_list =[]     
        for individual in y.diapause_list:  
                b_list.append(individual.b)
                c_list.append(individual.c)
                d_list.append(individual.d)
                e_list.append(individual.e)
                among_list.append(individual.among)
                within_list.append(individual.within)
                
        return([numpy.mean(b_list), numpy.mean(c_list), numpy.mean(d_list),
                numpy.mean(e_list), numpy.mean(among_list), numpy.mean(within_list)])
        
        
    def plot (self, to_plot = "surv_list"):
        '''docstring'''
        x_list = [i for i in range(len(self.result_list))]
        plt.plot(x_list, self.__dict__[to_plot], label = to_plot)
        plt.title(self.model_name)
        plt.legend()
        plt.show()
        
    
    def plot_results (self, to_plot = 0):
        y_list = []
        for i in self.result_list:
            y_list.append(i[to_plot])
        x_list = [i for i in range(len(self.result_list))]
        name = ["slope", "lower limit", "upper limit", "midpoint", "among", "within"]            
        plt.plot(x_list, y_list, label = name[to_plot])
        plt.legend()
        plt.show()

'''
test = Run_Program(growth_rate= 1.1, sigma_float = 0, max_year = 500, model_name = "evolving plasticity")
test.run()
for i in range(6):
    test.plot_results(i) # among is most important fig: high among = plasticity, low among = bet-hedging
#in test run this led to evolution of relatively flat slope (0.5) for first 500 years, but 
    #then slope increased again
'''

'''
test2 = Run_Program(growth_rate= 1.1, sigma_float = 4, max_year = 500, model_name = "evolving bet-hedging")
test2.run()
for i in range(6):
    test2.plot_results(i)
    #in test run this gave conservative bet-hedging (e = 14, c = 0, b >1), though 
    #towards the end d crashed (to 0.5), making it a very risk-prone bet-hedigng strategy
'''

'''
test3 = Run_Program(growth_rate= 1.1, sigma_float = 8, max_year = 1000, model_name = "extinction")
test3.run() #should eventually lead to extinction, but not crash
for i in range(6):
    test3.plot_results(i)

'''
   




'''
todo: 
    ***winter severity: low prob of dying when winter arrives --> canalization (loss of reaction norm)
    ***add a change of means with time.*** 
- mean increases at slow rate with year, sigma = 0
   ==> expectation: evolution of e with high plasticity (genetic tracking)
- mean increases at high rate, sigma =0
   ==> expectation: evolution of var_among (this essentially means that cue becomes too unreliable)
- mean increases at slow rate, sigma = 3
   ==>evolution of e, but also of lower plasticity
- mean increases at high rate, sigma =3
   ==> either faster evolution of high var_among (additive effects), or genetic assimilation:
    var_among evolves cyclically, e follows (in longer streaks of good years)
    
- sudden jump in mean, sigma = 0
==> extinction
- sudden jump in mean, sigma = 3
==>assimilation: evolution of bet-hedging ensures some flat slopes, these will then evovle back to high plasticity




 #mu = mu_float + climate_rate * i
        #---or----
        #if i > max_year/2:
            #mu = mu_float + climate_jump
            
        #y_list.append(Year(mu, sigma_float, popsize, eggs = y_list[i-1].diapause_list))
        
       # y_list.append(Year(mu_float, sigma_float, popsize, eggs = y_list[i-1].diapause_list))
        #diapausing eggs of last year become new year's population (and will be 
        #reduced to popsize at year initialization)) 
        
''' 