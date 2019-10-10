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
            new_b = self.b if random.uniform(0,1) > mut_frac else random.gauss(self.b,0.2)
            if new_b > 10:
                new_b = 10 #very large slopes cause math.range error in .var_comps()
                #if b*e > 710 => math.exp(710)
            new_c = self.c if random.uniform(0,1) > mut_frac else random.uniform(0,1)
            new_d = self.d if random.uniform(0,1) > mut_frac else random.uniform(new_c,1)
            new_e = self.e if random.uniform(0,1) > mut_frac else random.gauss(self.e,0.2)
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
        
    def var_comps(self):
        ''' calculate variance among environments (plasticity)
        
        var_within = sum p*(1-p) / n
        var_among = sd^2 / n'''
        
        probability = []
        p2 = []
        for t in range(50):
            upper = (self.d-self.c) #=upper part of diap_probability equation
            lower = 1 + math.exp(-1*self.b* (t-self.e)) #lower part
            prob = round(self.c + (upper/lower), 4)
            probability.append(prob)#note constant c
            p2.append(prob * (1-prob))
            #rounding included for numerical stability; sd(10^-130,10^-15, 0.2) etc
        self.among = numpy.std(probability, ddof=1)
        self.within = numpy.mean(p2)    
          
    
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
            else:
                offspring_list.extend(individual.reproduce(gr_float, mut_float))
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
        #print ("survivors: ", len(self.diapause_list))



            
    

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
        self.y_list = []    #stores Year instances
        
        pop_list = []
        
        for i in range(self.popsize):
            b = random.gauss(1,0.5)
            c = random.uniform(0,0.5)
            d= random.uniform(0.5,1)
            e = random.gauss(self.mu_float,self.sigma_float * 2)
            pop_list.append(Individual(b,c,d,e))
        init = Year(self.mu_float, self.sigma_float, self.popsize, pop_list)
        init.diapause_list = init.awake_list
        self.y_list.append(init)
        
    def __str__(self):
        return("PARAMETERS:\nr:{}N:{}mutrate:{}\n mu:{}sigma:{}end:{}".format(
                self.growth_rate, self.popsize,self.mut_rate,
                self.mu_float, self.sigma_float, self.max_year))
        
    def run(self):
        print ("Running")
        yc = 1
        for i in range(1, self.max_year): 
            self.y_list.append(Year(self.mu_float, self.sigma_float, self.popsize, 
                                    eggs = self.y_list[i-1].diapause_list))
            #diapausing eggs of last year become new year's population (and will be 
            #reduced to popsize at year initialization))
            if len(self.y_list[i].awake_list) == 0:
                print ("population extinct!")
                break
            self.y_list[i].runyear(self.growth_rate, self.mut_rate)
            yc +=1
            if not yc % 10: #if yc % 10 == 0
                print(".", end = '')
        self.y_list.pop(0)
        yc -= 1
        return(yc)
    
    def get_summary(self):
        '''provide winter onsets and #survivors'''
        for year in self.y_list:
            self.t_on_list.append(year.t_on)
            self.surv_list.append(len(year.diapause_list))
        

    def get_results(self, output = "b"):
        '''provide population level summary of reaction norm shape
    
        input: "b", "c", "d", "e", "among" or "within". any other input throws error'''
        result = []
        yc = 0
        for year in self.y_list:  
            yc += 1
            if not yc % 10:
                print("-",end="")
            ind_list = []
        
            for individual in year.diapause_list:  
                ind_list.append(individual.__dict__[output])
            result.append(numpy.mean(ind_list))
        self.__dict__[output+"_list"] = result
        return(result)
            
        
    def plot (self, to_plot = "surv_list", length =0):
        '''docstring'''
        x_list = [i for i in range(length)]
        plt.plot(x_list, self.__dict__[to_plot], label = to_plot)
        plt.title(self.model_name)
        plt.legend()
        plt.show()
        
    def var_comps(self):
        for year in self.y_list:
            for individual in year.diapause_list:
                individual.var_comps()

test = Run_Program(growth_rate= 1.1, sigma_float = 0, max_year = 1000, model_name = "evolving plasticity")
years = test.run() 
test.var_comps()
test.get_summary()
test.plot(length = years)
test.get_results(output = "among")
test.plot(to_plot= "among_list", length = years)



test2 = Run_Program(growth_rate= 1.1, sigma_float = 2, max_year = 1000, model_name = "evolving bet-hedging")
years2 = test2.run() 
test2.var_comps()
test2.get_summary()
test2.plot(length = years2)
test2.get_results(output = "among")
test2.plot(to_plot= "among_list", length = years2)

'''
'''
 #mu = mu_float + climate_rate * i
        #---or----
        #if i > max_year/2:
            #mu = mu_float + climate_jump
            
        #y_list.append(Year(mu, sigma_float, popsize, eggs = y_list[i-1].diapause_list))
        
       # y_list.append(Year(mu_float, sigma_float, popsize, eggs = y_list[i-1].diapause_list))
        #diapausing eggs of last year become new year's population (and will be 
        #reduced to popsize at year initialization))
        
'''      
y_list_bh = run_program(sigma_float = 3, max_year = 100)
print("bh done.\n\n")
res_bh = get_results(y_list_bh)
plot_details(res_bh,sig = 6)

y_list_p = run_program(sigma_float = 0)
print("plasticity done.\n\n")
res_p = get_results(y_list_p)
plot_details(res_p, sig=0)
summ = get_summary(y_list_bh)
plot_summary(summ)
'''

'''
todo: add a change of means with time. 
- mean increases at slow rate with year, sigma = 0
==> expectation: evolution of e with high plasticity (genetic tracking)
- mean increases at high rate, sigma =0
==> expectation: evolution of var_among (this essentially means that cue becomes too unreliable)
- mean increases at slow rate, sigma = 3
==>evolution of e, but also of lower plasticity
- mean increases at high rate, sigma =3
==> either faster evolution of high var_among (additive effects), or genetic assimilation:
    var_among evolves cyclically, e follows (in longer streaks of temporal autocorrelation)
    
- sudden jump in mean, sigma = 0
==> extinction
- sudden jump in mean, sigma = 3
==>assimilation: evolution of bet-hedging ensures some flat slopes, these will then evovle back to high plasticity


#results_list_bh = get_summary(y_list_bh)
#plot_summary(results_list)
plt.plot(x_list, b, label = "b" )
plt.plot(x_list, c, label = "c" )
plt.plot(x_list, d, label = "d" )
plt.plot(x_list, e, label = "e" )
'''