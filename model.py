# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
The model simulates the trade-off between investment in growth and in reproduction 
in unpredictable and changing conditions.
An individual is allowed to grow until it makes the irreversible decision to start 
dormancy, upon which its resources are converted into offspring. The longer it grows, 
the higher the number of offspring; but when winter arrives, all non-dormant individuals
die. If winter onset is unpredictable, there is a trade-off between high (arithmetic) 
mean growth rate and survival. 

A starting population has *popsize* individuals. Each individual has a different
genotype with four properties (*b,c,d,e* = slope, lower limit, upper limit, midpoint), 
which determine logistic reaction norm shape in response to *t* (time, e.g. day length). 
This reaction norm determines the probability of dormancy as function of time. 
While non-dormant, the individual grows linearly with a growth rate of *growth_rate*. Upon
deciding for dormancy, the resources are converted into offspring, at the rate
*growth_rate* \* *t*. The offspring inherit the same genotype, except for the possibilty 
of mutations with a rate of *mut_rate*. The offspring will remain dormant until the 
next year.
Winter onset *t_on* is drawn for a normal distribution with mean *mu_float* and 
standard deviation *sigma_float*. When *t* reaches *t_on*, the growth rate 
is set to 0 (except for models that include winter severity, see below), i.e. the 
individual dies. When *t* reaches *t_max* the year is over and the seed bank 
replaces the population. If the seed bank is larger than *popsize*, 
*popsize* individuals are randomly drawn from the seed bank.
If winter onset is predictable (standard deviation = 0), the reaction
norm is expected to evolve a steep slope (plasticity), inducing dormancy just before 
mean winter onset. If the standard deviation is high (winter unpredictable), the
reaction norm may evolve an early inflection point (conservative bet-hedging), 
or a flat slope, such that some offspring will be diapausing regardless of winter onset
(diversified bet-hedging). The shape of the reaction norm can be summarised by variance among 
environments and variance within environments (see also  https://doi.org/10.1101/752881 and 
10.32942/osf.io/trg34).
While all individuals are expected to die at *t_on*, there is some probability that the 
winter is not severe and that the individual can continue reproduction throughout the season.
This probability is adjusted by the model parameter *severity*. Decreasing winter severity 
is expected to increase canalization, i.e. a decreased upper limit of the reaction norm shapes

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
    '''creates individual with 4 reaction norm properties.
    
    reaction norm is given by p(diap) = c + (d-c)/(1+math.exp(-b*(t-e))) with b =slope,
    c,d = lower and upper limit, e = inflection point'''
    
    def __init__(self,b,c,d,e): #all float
        self.b = b #-inf to +inf, though values > |5| carry little meaning
        self.c = c #0 to 1
        self.d = d #0 to 1, though should generally be >c
        self.e = e #- inf to +inf, expected ~ 25
        
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
        for i in range(0,integ): 
            n = Individual(self.b, self.c, self.d, self.e)
            n.mutate(mut_frac)
            #for r =3.5, the loop will run 2 times in 50% of cases, 3 times in the other 50%
            return_list.append(n)
        return (return_list)
    
    def check_diap (self, t_int):
        '''test for diapause given individual's reaction norm shape and t'''
        diap_probability = self.c + (self.d-self.c)/(1+math.exp(-self.b*(t_int-self.e))) 
        diap_bool = bool(numpy.random.binomial(1,diap_probability))
        return(diap_bool)
        
    def var_comps(self, t_max):
        ''' calculate variance among and within environments
        
        var_within = sum p*(1-p) / n
        var_among = sd^2 / n'''
        
        probability = []
        p2 = []
        for t in range(t_max):
            upper = (self.d-self.c) #=upper part of diap_probability equation
            if (t - self.e >50): #this check is needed to prevent math.range error
                lower = 1
            else:
                lower = 1 + math.exp(-1*self.b* (t-self.e)) #lower part
            prob = round(self.c + (upper/lower), 4)
            probability.append(prob)
            p2.append(prob * (1-prob))
            
        self.among = numpy.std(probability, ddof=1)
        self.within = numpy.mean(p2)    
        
    def mutate(self, mut_frac):
       self.b = self.b if random.uniform(0,1) > mut_frac else \
            min(random.gauss(self.b,0.1),10) #min makes sure value stays below 10
            #b must be < 10, because very steep slopes cause math.range error in .var_comps()
            #math.exp(b*e) can become too large to handle
       self.c = self.c if random.uniform(0,1) > mut_frac else \
            min(max(0,random.gauss(self.c,0.02)),1) # 0< c < 1
       self.d = self.d if random.uniform(0,1) > mut_frac else \
            min(max(self.c,random.gauss(self.d,0.02)),1)     #c<d<1
       self.e = self.e if random.uniform(0,1) > mut_frac else \
            random.gauss(self.e,0.1)
    

class Run_Program(object):
    '''run the model'''
    def __init__ (self, growth_rate = 0.5, popsize = 500, summer_mut = 0, winter_mut = 1/100,
                max_year = 20000, t_max = 50, model_name = "generic", winter_list = [],
                severity = 1, startpop = []):
        '''saves parameters and creates starting population'''
        
        #parameters
        self.growth_rate = growth_rate #no offspring for each time step in which
        #the individual is not dormant; e.g. 0.5, dormancy at day 25 --> 12.5 offspring
        self.popsize = popsize #population size at start of the year
        self.summer_mut = summer_mut #not implemented, leave at 0
        self.winter_mut = winter_mut 
        self.max_year = max_year
        self.t_max = t_max
        self.model_name = model_name
        self.winter_list = winter_list
        self.severity = severity #penalty of not being dormant after winter arrival 
        self.results = numpy.zeros((self.max_year,10))
        self.all_results = []
        #stores: w_on; no. survivors; b; c; d; e; within; among, ratio, sum
        
        #starting population
        self.eggs = startpop
        if not self.eggs:
            for i in range(self.popsize):
                b = random.gauss(1,3)
                c = random.uniform(0,0.5)
                d= random.uniform(0.5,1)
                e = random.gauss(25, 5)
                self.eggs.append(Individual(b,c,d,e))
                
        if not winter_list:
            self.winter_list = [(self.t_max/2) for i in range(self.max_year)]
        
    def __str__(self):
        first_line = "PARAMETERS\n"
        second_line = str(" r: {:5.2f}      N:{:4}    summer mut: {:4.2f} \n".format(
                self.growth_rate, self.popsize,self.summer_mut))
        third_line = str("winter mut: {:4.2f}    end: {:8}".format(self.winter_mut, self.max_year))
        return(first_line+second_line+third_line)
        
        
    def run(self):
        print("\nmodel: ", self.model_name, "\nRunning. 1 dot = 100 years")
        yc = 1
          
        for i in range(0, self.max_year): 
            w_on = self.winter_list[i]
            awake_list = self.eggs if len(self.eggs) < self.popsize else\
            random.sample(self.eggs, self.popsize)   
            if not awake_list:
                print ("population extinct!")
                yc+=1
            else:
                for individual in self.eggs:
                    individual.mutate(self.winter_mut)
                self.eggs = self.runyear(awake_list, w_on)        
           #     self.results[i,:] = self.save_results(w_on, self.eggs)   
                self.all_results.append(self.save_all(self.eggs))
                yc +=1
                if not yc % 100: 
                    print(".", end = '')
        return(yc)
        
    def runyear(self, curr_list, end):
        '''on each day, every Individual may enter diapause; if awake, it reproduces
        
        input: list of individuals; number of days until winter onset
        output: offspring (from those indivdiduals that made it to dormancy before winter'''
        diapause_list = []
        severe_bool = bool(numpy.random.binomial(1,self.severity))
        for t in range(self.t_max):
            gr = self.growth_rate * t # if t < end else self.growth_rate * (1-self.severity) * t  
            newlist = []
            if (curr_list and not (t > end and severe_bool)):
                for individual in curr_list:
                    if individual.check_diap(t):
                        diapause_list.extend(individual.reproduce(gr, self.summer_mut))
                    else: 
                        newlist.append(individual)
            curr_list = newlist
        return(diapause_list)      

    
    def save_all (self, eggs):
        x = numpy.zeros((len(eggs),3))
        t= 0
        for individual in eggs:
            x[t,:] = [round(individual.b*3)/3, round(individual.d*20)/20, 
             round(individual.e)]
            t = t+ 1

        b_un = numpy.unique(x[:,0], return_counts=  True)
        d_un = numpy.unique(x[:,1], return_counts=  True)
        e_un = numpy.unique(x[:,2], return_counts=  True)
        result = [b_un[0], d_un[0], e_un[0], b_un[1], d_un[1], e_un[1]]
        return (result)

    def plot_all (self, to_plot=0):
        lablist = ["slope", "upper limit", "midpoint", "survivors"]
        ax = [8,1,40,0] 
        x = []
        y = []
        if to_plot <3:
            for i in range(len(self.all_results)):
                x.append (self.all_results[i][to_plot])
                y.append (self.all_results[i][to_plot+3])
        else:
            for i in range(len(self.all_results)):
                x.append(sum(self.all_results[i][3]))
            ax[3] = max(x)
        plt.plot([])
        plt.ylabel(lablist[to_plot])
        plt.title(self.model_name)
        plt.axis([0, len(self.all_results), 0 ,ax[to_plot]])
        if to_plot <3:    
            for i in numpy.arange(0,len(self.all_results),int(self.max_year/100)):
                plt.scatter(numpy.full((len(x[i])),i), x[i], c = y[i]/sum(y[i]), 
                        cmap = "Blues_r", s =10, marker = "s")
        else:
            plt.plot(x)
        plt.show()


#not used anymore:
    def save_results(self, w_on, eggs):
        '''stores: w_on; no. survivors; b; c; d; e; among; within'''
        x = numpy.zeros((len(eggs),8))
        t=0

        for individual in eggs:
            individual.var_comps(self.t_max)
            x[t,:] = [individual.b, individual.c, individual.d, 
             individual.e,
             individual.among, individual.within, 
             individual.among/(individual.among+individual.within), 
             individual.among + individual.within]
            t = t + 1
        return (numpy.concatenate(([w_on], [len(eggs)],numpy.mean(x,axis=0)),axis=0))
            
    def plot (self, to_plot = 0):
        '''docstring'''
        lablist = ["Winter onset", "No. survivors", "slope", "Lower limit", 
                   "Upper limit", "Midpoint", "Variance among", "Variance within",
                   "Variance ratio", "Phenotypic variance"]
        plt.plot(self.results[:, to_plot])
        plt.ylabel(lablist[to_plot])
        plt.title(self.model_name)
        plt.show()



'''functions'''
def make_pop(model, popsize):
    '''builds a population based on final population of another run'''
    pop = []
    r =  random.sample(range(len(model.eggs)), popsize)   
    for i in range(popsize):
        pop.append(Individual(b = model.eggs[r[i],2], 
                      c = model.eggs[r[i],3],
                      d = model.eggs[r[i],4],
                      e = model.eggs[r[i],5]))
    return(pop)


def make_climate(mu =25, sigma = 0, trend = 0, n = 20000):
    '''creates climate data'''
    climate =[]
    n = int(n)
    for i in range(n):
        climate.append(round(random.normalvariate(mu,sigma)) + trend*i)
    return(climate)
    
    
my = 10    #approx 9 sec/100 years = 30 min per model
var_climate = make_climate(sigma = 3, n = my)

climate_slow = make_climate(trend = 1/10, n = my/2)
climate_fast = make_climate(trend =  1/2, n = my/4)
climate_neg  = make_climate(trend = -1/2, n = my/4, mu = my/8)
climate_slow_var = make_climate(sigma = 3, trend = 1/10, n = my/2)
climate_fast_var = make_climate(sigma = 3, trend =  1/2, n = my/4)
climate_neg_var  = make_climate(sigma = 3, trend = -1/2, n = my/4, mu = my/8)

'''standard models '''          
predictable= Run_Program(model_name = "Predictable climate", max_year = my)
predictable_mild = Run_Program(model_name = "Predictable but mild", max_year = my,
                   severity = 0.5)

variable = Run_Program(model_name = "Unpredictable",  max_year = my, 
                       winter_list = var_climate)
variable_mild = Run_Program(model_name = "Unpredictable and mild", max_year = my, 
                   winter_list = var_climate, severity = 0.5)

predictable.run()
predictable_mild.run()
variable.run() 
variable_mild.run()


'''plastic genotype in variable climate and vice versa'''
plast_pop = make_pop(predictable, 500)
var_pop = make_pop(variable, 500)

plast_in_var = Run_Program(model_name = "plastic genotype, variable env", max_year = my,
                           winter_list =var_climate, startpop = plast_pop)
var_in_plast = Run_Program(model_name = "bet-hedger, stable env", max_year = my,
                           startpop = var_pop)

plast_in_var.run()
var_in_plast.run()
 


'''trends in environment'''
slow_plastic = Run_Program(model_name = "plastic +slow", my = my/2, 
                   winter_list = climate_slow, startpop = plast_pop, 
                   t_max =my/2)
fast_plastic = Run_Program(model_name = "plastic + fast", max_year =my/4,
                   winter_list = climate_fast, startpop = plast_pop,
                   t_max = my/4)
neg_plastic = Run_Program(model_name = "plastic + negative", 
                   winter_list = climate_neg, max_year = my/4, 
                   startpop = plast_pop, t_max = my/4)
slow_var = Run_Program(model_name = "bet-hedger + slow", max_year = my/2,
                   winter_list = climate_slow, startpop = var_pop, t_max = my/2)
fast_var = Run_Program(model_name = "bet-hedger + fast",  max_year = my/4, 
                   winter_list = climate_fast, startpop = var_pop,t_max = my/4)
neg_var = Run_Program(model_name = "bet-hedger + negative", max_year = my/4, 
                   winter_list = climate_neg, startpop = var_pop,t_max=my/4)

slow_plastic.run()
fast_plastic.run()
neg_plastic.run()
slow_var.run()
fast_var.run()
neg_var.run()


climate_step = [25 if i<5 else 20 for i in range(my/4)]
step_plastic = Run_Program(model_name = "step + plastic", max_year = my/4,
                   winter_list = climate_step, startpop = plast_pop)
step_bh      = Run_Program(model_name = "step + bet-hedger",max_year = my/4,
                   winter_list = climate_step, startpop = var_pop) 

step_plastic.run()
step_bh.run()






'''
climate = []
file = open('E:\g02-modelling\AGE00147705.txt', 'r')
line = file.readline()
while line:
    climate.append(round(int(line.strip())/8))
    line = file.readline()
file.close()
clim2 = random.sample(climate, 5000)
test = Run_Program(growth_rate = 1.05, winter_list = clim2, max_year= 5000, model_name = "test")
#not tested after changing code a bit, but older run results: 
#no survivors ~ 1500-200 with frequent drops to ~ 500; slope 1-3, upper limit 0.6 - 0.9,
#slope negatively correlated with upper lim, midpoint slow increase 40-43'''
 