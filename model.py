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
        x = str("slope: {:.3f} lower limit: {:.3f} upper limit: {:.3f} midpoint: {:.3f}".format(
                self.b, self.c, self.d, self.e))
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
            new_b = self.b if random.uniform(0,1) > mut_frac else random.gauss(self.b,0.1)
            new_c = self.c if random.uniform(0,1) > mut_frac else random.gauss(self.c,0.1)
            new_d = self.d if random.uniform(0,1) > mut_frac else random.gauss(self.d,0.1)     
            new_e = self.e if random.uniform(0,1) > mut_frac else random.gauss(self.e,0.1)
            
            #some exceptions: c has to be in range {0,1}, d in range {c,1} 
            #b must be < 10, because very steep slopes cause math.range error in .var_comps()
            #math.exp(b*e) can become too large to handle
            if new_b > 10:
                new_b = 10 
            if new_c < 0:
                new_c = 0
            if new_c > 1:
                new_c = 1
            if new_d < new_c: 
                new_d = new_c
            if new_d > 1:
                new_d = 1  
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
          
    
class Year(object):
    '''decisions of diapause and reproduction for 1 year
    
    input: mu = mean winter onset, sigma = sd winter onset, both float; eggs = list
    of instances of class Individual'''
    def __init__ (self, mu, sigma, popsize, eggs):
        self.t_on = random.normalvariate(mu, sigma)
        self.t_on = round(self.t_on)
        self.awake_list = eggs if len(eggs) < popsize else random.sample(eggs, popsize)
        #this winter carrying capacity is required to keep population sizes in check;
        #summer carrying capacities would reduce the need to use the whole season
        #(diminishing payoffs) and select for conservative reaction norms
        self.diapause_list = []
    
    def __str__(self):
        return ("Winter onset on day {}\nIn diapause:{}; awake:{}".format(
                self.t_on, len(self.diapause_list),len(self.awake_list)))
    
    def runday(self, growth_rate, mut_rate, t):
        '''on each day, every Individual may enter diapause; if awake, it reproduces
        
        input: growth rate, mutation rate, both float; and t(int)'''
        offspring_list= []
        remove_from_awake_list =[]
        for individual in self.awake_list:
            if individual.check_diap(t):
                individual.var_comps() #time-intensive, so only performed when needed
                self.diapause_list.append(individual)
                remove_from_awake_list.append(individual)
            else:
                offspring_list.extend(individual.reproduce(growth_rate, mut_rate))
                #extend instead of append here because return of reproduce is a list, not
                #an individual instance of Individual
        for rem in remove_from_awake_list:
            self.awake_list.remove(rem)
        self.awake_list.extend(offspring_list)
    
    def runyear(self, growth_rate, mut_rate):
        '''do daily stuff until winter arrives; then kill all non-diapausing Individuals'''
        
        for t in range(self.t_on):
            self.runday(growth_rate, mut_rate, t)
        #now that winter has arrived, self.awake_list could be discarded.
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
        self.result_list =[]
        self.old = self.initialize_pop()    #stores Year instances
    
    def initialize_pop(self):
        pop_list = []
        for i in range(self.popsize):
            b = random.gauss(1,0.5)
            c = random.uniform(0,0.5)
            d= random.uniform(0.5,1)
            e = random.gauss(self.mu_float,self.sigma_float * 2)
            pop_list.append(Individual(b,c,d,e))
        init = Year(self.mu_float, self.sigma_float, self.popsize, pop_list)
        init.diapause_list = init.awake_list
        return(init)
        
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
        for i in range(1, self.max_year): 
            y = Year(self.mu_float, self.sigma_float, self.popsize, 
                                    eggs = self.old.diapause_list)
            #this line also implies reduction to popsize at Year initialization
            if len(y.awake_list) == 0:
                print ("population extinct!")
                yc+=1
                break
            y.runyear(self.growth_rate, self.mut_rate)
            self.result_list.append(self.save_all_results(y))
            self.t_on_list.append(y.t_on)
            self.surv_list.append(len(y.diapause_list))
            yc +=1
            if not yc % 10: #if yc % 10 == 0
                print(".", end = '')
            self.old = y
        return(yc)

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

test = Run_Program(growth_rate= 1.1, sigma_float = 0, max_year = 500, model_name = "evolving plasticity")
test.run()
for i in range(6):
    test.plot_results(i)
#may lead to evolution of relatively flat slope (0.5) for first 500 years, next few
    #thousand years leaad back to steep slope

'''
test2 = Run_Program(growth_rate= 1.1, sigma_float = 2, max_year = 500, model_name = "evolving bet-hedging")
test2.run()
for i in range(6):
    test2.plot_results(i)
    #looks like risk-prone strategy(d becomes ~0.5), at least for first 500 runs. comparable to halketts model
'''

'''
test3 = Run_Program(growth_rate= 1.1, sigma_float = 8, max_year = 1000, model_name = "extinction")
test3.run()
for i in range(6):
    test3.plot_results(i)


discovered bug: when sigma is high and populations get low, y_list sometimes 
stores wrong values. Survival figure shows in some years pop size is 
reported as 0, but program keeps running. print statement in test.run method
shows that pop size does not actually decrease to zero, so saving in y_list is 
bugged '''
    




'''
todo: 
    ***add "all" methods to save_results
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