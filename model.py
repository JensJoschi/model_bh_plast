# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

''' to do list
directional change environment( ind.redirect())
currently a flat reaction norm at 0.1 means that probability to be NOT in diapause 
is 0.9; 0.9^2; 0.9^3... 0.9^t --> even with flat reaction norm cumulative probability goes 
towards 1 quickly. this is unrealistic, because a multivoltine organism gets only ~ 5 decisions 
to diapuase, not 50
testing!
check save and output functions
'''

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
import copy
import os

'''classes'''
class Individual(object):
    ''' same as class individuum, but reaction norm not restricted to logit shape'''
    
    def __init__(self,p_list = []):
        self.p_list = p_list if p_list else [
                numpy.random.uniform(0,0.1) for i in range(50)]
        
    def __str__(self):
        return(str(self.p_list)) 
    
    def reproduce(self, r): 
        '''reproduces the individual r times
        
        r = 4.2 will return a list with 4 individuals in 80% of all cases, and with 5
        individuals in the remaining 20% of all cases. expected input: r = float > 0'''
        n = math.floor(r) + numpy.random.binomial(1, r- math.floor(r),1)[0]
        return( [copy.deepcopy(self) for i in range(0,n)] )  
             
    def mutate(self, mut_frac):
       '''induces mutations in the individual's reaction norm'''
         
       self.p_list = [i if (random.uniform(0,1) > mut_frac) else 
                     random.choice([0,random.uniform(0,1),1]) for i in self.p_list]
       #mutations make each loci 0, 1, or a random draw between 0 and 1
    def cumprob (self):
        x = numpy.cumprod([1-i for i in self.p_list]) #cumulative probability to be NOT in diapause
        return([1-i for i in x]) #cum prob to be in diapause
        
    def pars(self):
        ''' calculate variance among and within environments

        is done on cumulative probability of being in diapause
        var_within = sum p*(1-p) / n
        var_ax.mong = sd^2 / n'''
        p = self.cumprob()
        p2 =[i * (1-i) for i in p]
 
        i = 0
        mp = False
        while not mp:
            if p[i] > max(p)/2:
                mp = i
            i = i + 1
        among = numpy.std(p, ddof=1)
        within = numpy.mean(p2) 
        return([mp, among, within]) 
    
    def redirect(self):
        #will eventually be used for directional change in environment
        pass

 

class Ind_logit (Individual):
    '''creates individual with 4 reaction norm properties.
    
    reaction norm is given by p(diap) = c + (d-c)/(1+math.exp(-b*(t-e))) with b =slope,
    c,d = lower and upper limit, e = inflection point'''
    
    def __init__(self, b=[], c=[], d =[], e = [], t = 50): #all float
        self.b = random.uniform(0,10) if b == [] else b
        self.c = random.uniform(0,1)  if c == [] else c
        self.d = random.uniform(self.c,1)if d == [] else d 
        self.e = random.uniform(0,50) if e == [] else e
        
        self.p_list = []
        for i in range(t): #calculate diapause given b,c,d,e
            exponent = max(min(-self.b * (i - self.e), 500),-500)
        # extreme values can occur for t >1000 in models using climate_neg
            self.p_list.append(self.c + (self.d-self.c)/(1+math.exp(exponent)))
        
    def __str__ (self):
        x = f"slope: {self.b:.3f} lower limit: {self.c:.3f} upper limit: {self.d:.3f} midpoint: {self.e:.3f}"
        return (x)
     
    def mutate(self, mut_frac):
       '''induces mutations in the individual's reaction norm'''
       self.b = self.b if random.uniform(0,1) > mut_frac else \
            random.uniform(0,10) #min makes sure value stays below 10
            #b must be < 10, because very steep slopes cause math.range error in .var_comps()
            #math.exp(b*e) can become too large to handle
       self.c = self.c if random.uniform(0,1) > mut_frac else \
            random.uniform(0,1) # 0< c < 1
       self.d = self.d if random.uniform(0,1) > mut_frac else \
            random.uniform(self.c,1)     #c<d<1
       self.e = self.e if random.uniform(0,1) > mut_frac else \
            random.uniform(0,50)

        
        
class Run_Program(object):
    '''run the model'''
    def __init__ (self, model_name = "generic", max_year = 10000, winter_list = [],
                  severity = 1, startpop = [], direction = 0, saving =True, cont = True,
                  popsize = 1000, winter_mut = 1/10000, t_max = 50, growth_rate = 0.8):
        '''saves parameters and creates starting population'''
        
        self.model_name = model_name
        self.max_year = max_year
        self.winter_list = winter_list
        self.severity = severity #penalty if not being dormant after winter arrival 
        self.startpop = startpop
        self.direction  = direction #climate change in days/year.
        #this will actually not change the climate data, but the mean of the reaction 
        #norms of all individuals
        self.saving = saving
        self.popsize = popsize #population size at start of the year
        self.winter_mut = winter_mut 
        self.t_max = t_max
        self.growth_rate = growth_rate #no offspring for each time step in which
        #the individual is not dormant; e.g. 0.5, dormancy at day 25 --> 12.5 offspring
        self.model = "Individual" if cont else "Ind_logit"
        self.fitness_list = []
        self.details_list = []
        
        self.eggs = startpop
        if not self.eggs:
            self.eggs = [eval(self.model+"()") for i in range(self.popsize)]
            #produces popsize individuals, either of class Individual or of class Ind_logit
                
        if not winter_list:
            self.winter_list = [(self.t_max/2) for i in range(self.max_year)]
        print("Initialized model '", model_name, "'")
        
    def __str__(self):
        first_line = str("model: {}\n".format(self.model_name))
        second_line = str("Years: {:6}   N:{:4}   severity: {:4.2f}\n".format(
                self.max_year, self.popsize, self.severity))
        third_line = str("r: {:5.2f},  season length: {:3}, mutation rate: {:4.2f}\n".format(
                self.growth_rate, self.t_max, self.winter_mut))
        return(first_line+second_line+third_line)
        
        
    def run(self):
        print("\nmodel: ", self.model_name, "\nRunning. 1 dot = 100 years")
        yc = 1
        extinct_at = []
          
        for i in range(0, self.max_year): 
            w_on = self.winter_list[i]
            awake_list = self.eggs if len(self.eggs) < self.popsize else\
            random.sample(self.eggs, self.popsize)   
            if awake_list:
                for individual in self.eggs:
                #    individual.redirect(self.direction)
                    individual.mutate(self.winter_mut)
                self.eggs = self.runyear(awake_list, w_on)        
                self.fitness_list.append(len(self.eggs))
                if self.saving:
                    self.details_list.append(self.save_details())
                yc +=1
                if not yc % 100: 
                    print(".", end = '')
            elif extinct_at:
                pass
            else:
                extinct_at = yc
        if extinct_at:
            print ("Population extinct in year", extinct_at)
        return(yc)
        
    def runyear(self, curr_list, end):
        '''on each day, every Individual may enter diapause; if awake, it reproduces
        
        input: list of individuals; number of days until winter onset
        output: offspring from those indivdiduals that made it to dormancy before winter'''
        diapause_list = []
        severe_bool = bool(numpy.random.binomial(1,self.severity)) #is winter severe?
        #if winter is not severe, it does not have an effect of life history
        for t in range(self.t_max):
            gr = self.growth_rate * t 
            newlist = []
            if (curr_list and not (t > end and severe_bool)):#if there are non-diapausing
                #individuals left, and winter either did not arrive yet or is not severe:
                for individual in curr_list:
                    diapausing = bool(numpy.random.binomial(1,individual.p_list[t]))
                    if diapausing:
                        diapause_list.extend(individual.reproduce(gr))
                    else: 
                        newlist.append(individual)
            curr_list = newlist
        if not severe_bool: #make sure everyone who survives gets to diapause at end of season
            diapause_list.extend([individual.reproduce(gr) for individual in curr_list])
        return(diapause_list)      

    
    def save_details (self):
        eggs = random.sample(self.eggs,min(len(self.eggs),100))
        x = [individual.pars() for individual in eggs]
        mp =[]
        among=[]
        within=[]
        for i in x:
            mp.append(i[0])
            among.append(round(i[1],2))
            within.append(round(i[2],2))

        mp_un = numpy.unique(mp, return_counts=  True)
        among_un = numpy.unique(among, return_counts=  True)
        within_un = numpy.unique(within, return_counts=  True)
 
        result = [mp_un[0], among_un[0], within_un[0], 
                  mp_un[1], among_un[1], within_un[1]]
        return (result)

    def plot_all (self, to_plot=0):
        lablist = ["midpoint","among", "within", "fitness"]
        ax = [50,1,1,0] 
        x = []
        y = []
        if to_plot <3:
            for i in range(len(self.details_list)):
                x.append (self.details_list[i][to_plot])
                y.append (self.details_list[i][to_plot+3])
        else:
            x = self.fitness_list
            ax[3] = max(x)
        fig = plt.plot([])
        plt.ylabel(lablist[to_plot])
        plt.title(self.model_name)
        plt.axis([0, len(self.fitness_list), 0 ,ax[to_plot]])
        if to_plot <3:    
            for i in numpy.arange(0,len(self.fitness_list),int(self.max_year/100)):
                plt.scatter(numpy.full((len(x[i])),i), x[i], c = sum(y[i])/y[i], 
                        cmap = "Blues_r", s =10, marker = "s")
        else:
            plt.plot(x)
        plt.close()
        return(fig)


    def save_data(self):
        os.mkdir(self.model_name)
        numpy.save(os.path.join(os.getcwd(), self.model_name, self.model_name),
                       self.fitness_list) #model name twice to make generic.npy in subfolder "generic"
        numpy.save(os.path.join(os.getcwd(), self.model_name, "climate"), 
                       self.winter_list)
        numpy.save(os.path.join(os.getcwd(), self.model_name, "startpop"), 
                       self.startpop)
        fig = self.plot_all(3)
        fig[0].figure.savefig(os.path.join(os.getcwd(), self.model_name, 
                 "fitness.png"))
        if self.saving:
            numpy.save(os.path.join(os.getcwd(), self.model_name, self.model_name),
                       self.details_list)
            for i in range(3):
                fig = self.plot_all(i)
                fig[0].figure.savefig(os.path.join(os.getcwd(), self.model_name, 
                   str(i) + ".png"))
                

'''functions'''
def make_pop(model, popsize):
    '''builds a population based on final population of another run'''
    pop = []
    r =  random.sample(range(len(model.eggs)), popsize)   
    for i in range(popsize):
        pop.append(Individual(b = model.eggs[r[i]].b, 
                      c = model.eggs[r[i]].c,
                      d = model.eggs[r[i]].d,
                      e = model.eggs[r[i]].e))
    return(pop)


def make_climate(mu =25, sigma = 0, trend = 0, n = 20000):
    '''creates climate data'''
    climate =[]
    n = int(n)
    for i in range(n):
        climate.append(round(random.normalvariate(mu,sigma)) + trend*i)
    return(climate)

    
plasticity = Run_Program(max_year = 1000, saving =False, model_name ="plastic") 
plasticity.run()

print ("bet-hedging.")
var_climate = make_climate(sigma = 4, n = 10000)
bet_hedging = Run_Program(max_year = 10000, saving =False, winter_list =var_climate,
                          model_name = "bh")
#bet_hedging.run()


'''
p_mild = Run_Program(max_year =2000, saving =False, severity = 0.3, model_name = "p_mild")
p_mild.run()

bh_mild = Run_Program(max_year = 2000, saving =False, winter_list =var_climate,
                      severity = 0.3, model_name = "bh_mild")
bh_mild.run()
'''


'''environments and climates for further models'''
#plast_pop = make_pop(plasticity, 1000)
#var_pop = make_pop(bet_hedging, 1000)

 

'''trends in environment'''
trend = [-0.1, -0.01, 0, 0.01, 0.1]
'''
for i in range(len(trend)):
    Run_Program(model_name = "plastic_predictable_"+str(trend[i]),
                startpop = plast_pop, direction = trend[i]).run()
    Run_Program(model_name = "bh_predictable_"+str(trend[i]),
                startpop = var_pop, direction = trend[i]).run()
    Run_Program(model_name = "plastic_unpredictable_"+str(trend[i]),
                startpop = plast_pop, direction = trend[i],
                winter_list= var_climate).run()
    Run_Program(model_name = "bh_unpredictable_"+str(trend[i]),
                startpop = var_pop, direction = trend[i],
                winter_list= var_climate).run()
'''