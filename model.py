# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:45:18 2019

@author: Jens
"""

'''
In this model we consider a multivoltine haploid genotype that faces the decision 
to either produce another generation or to diapause. The timing of the decision 
is randomized across individuals and years (uniform distribution from day 1 to 10), 
thus assuming that life cycles of individuals are not synchronized to each other. 
Individuals that choose to diapause gain less offspring than those that remain active, 
but individuals that do not diapause when winter arrives forfeit their opportunity
to produce offspring at all (except if winter is not severe, see below). The generation
time is fixed at t = 10 days, and winter onset occurs on average on day 15. Each
individual has 10 properties (*p_list*), which determine the reaction norm shape
of diapause. Winter onset *t_on* is drawn for a normal distribution with mean 15 
and standard deviation *sigma_float*. 

The offspring inherit the same genotype as their mother, except for the possibility 
of mutations with a rate of *mut_rate*. The offspring will remain 
dormant until the next year.
When the year is over the seed bank replaces the population. If the seed bank is 
larger than *popsize*, *popsize* individuals are randomly drawn from the seed bank. 
The growth rate is fixed at 1.2 for diapausing individuals, and 1.4. for individuals 
that produce another generation in case that the generation is complete before winter 
arrives. The fitness loss for non-dormant individuals upon winter onset ranges 
from 0 – 100 % and is adjusted by the model parameter *severity*.
A starting population has *popsize* individuals. The model runs for x years, 
after which population size and reaction norm parameters (Var_among, var_within,
mean) are recorded. 

explanation of var among + within needed. ref to preprint as soon as available
'''


import random
import numpy
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
import copy
import os

'''classes'''
class Individual(object):
    ''' Creates an individuum with a reaction norm shape determined by 10 parameters'''
    
    def __init__(self,p_list = []):
        self.p_list = p_list if p_list else [
                numpy.random.uniform(0,1) for i in range(10)]
        
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
    def diapause (self, t):
        return(bool(numpy.random.binomial(1,self.p_list[t],1)[0]))
        
    def pars(self):
        ''' calculate variance among and within environments

        var_within = sum p*(1-p) / n
        var_ax.mong = sd^2 / n'''
        p = self.p_list
        p2 =[i * (1-i) for i in p]
 
        i = 0
        mp = "False" #gets midpoint of the reaction norm
        while mp == "False": # better would be "while not mp" but this causes a bug if mp becomes 0
            if p[i] >= max(p)/2:
                mp = i
            i = i + 1
        among = numpy.std(p, ddof=1)
        within = numpy.mean(p2) 
        return([mp, among, within]) 
 
        
        
class Run_Program(object):
    '''run the model'''
    def __init__ (self, model_name = "generic", max_year = 10000,saving =True, 
                  winter_list = [], startpop = [], 
                  popsize = 1000, mut_rate = 1/10000, 
                  gr_diap = 1.2, gr_awake = 1.4, severity = 1):
        '''saves parameters and creates starting population'''
        
        self.model_name = model_name
        self.max_year = max_year
        self.saving = saving
        self.winter_list = winter_list
        self.startpop = startpop
        self.popsize = popsize #population size at start of the year
        self.mut_rate = mut_rate
        self.gr_diap = gr_diap #growth rate if deciding to diapause
        self.gr_awake = gr_awake #growth rate if deciding to stay awake, and if winter 
                    #is not there yet
        self.gr_loss = gr_diap * (1 - severity) #penalty if not being dormant after winter arrival 
        
        self.t_max = 30
        self.fitness_list = []
        self.details_list = []
        
        self.eggs = startpop
        if not self.eggs:
            self.eggs = [Individual() for i in range(self.popsize)]
            #produces popsize individuals
                
        if not winter_list:
            self.winter_list = [(self.t_max/2) for i in range(self.max_year)]
        print("Initialized model '", model_name, "'")
        
    def __str__(self):
        first_line = str("model: {}\n".format(self.model_name))
        second_line = str("Years: {:6}   N:{:4}   mutrate: {:4.2f}\n".format(
                self.max_year, self.popsize, self.mut_rate))
        third_line = str("r: {:5.2f},{:5.2f},{:5.2f},  season length: {:3}\n".format(
                self.gr_diap, self.gr_awake, self.gr_loss, self.t_max))
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
                    individual.mutate(self.mut_rate)
                self.eggs = self.runyear(awake_list, w_on)        
                self.fitness_list.append(len(self.eggs))
                if self.saving:
                    self.details_list.append(self.save_rn(
                            random.sample(self.eggs,min(len(self.eggs),100))
                            ))
                yc +=1
                if not yc % 100: 
                    print(".", end = '')
            elif extinct_at:
                pass
            else:
                extinct_at = yc
        if extinct_at:
            print ("Population extinct in year", extinct_at)
        self.results = numpy.array(self.save_rn(self.eggs))
        if self.saving:
            self.details_list = numpy.array(self.details_list)
        return(yc)
        
    def runyear(self, curr_list, winter):
        '''on each day, every Individual may enter diapause; if awake, it reproduces
        
        input: list of individuals; number of days until winter onset
        output: offspring from those indivdiduals that made it to dormancy before winter'''
        survivor_list = []
        for individual in curr_list:
            #make diapause decision based on t
            t = random.choice(range(10))
            if individual.diapause(t): 
              off = self.gr_diap
            else:
                if t < (winter - 10):
                    off = self.gr_awake
                else:
                    off = self.gr_loss
            survivor_list.extend(individual.reproduce(off))
        return(survivor_list)

    
    def save_rn (self, eggs):
        x = [individual.pars() for individual in eggs]
        mp =[]
        among=[]
        within=[]
        for i in x:
            mp.append(i[0])
            among.append(round(i[1],2))
            within.append(round(i[2],2))
        return([mp, among, within])
   
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


def make_climate(mu =15, sigma = 0, trend = 0, n = 20000):
    '''creates climate data'''
    climate =[]
    n = int(n)
    for i in range(n):
        climate.append(round(random.normalvariate(mu,sigma)) + trend*i)
    return(climate)

def plot_3d (rn):
       m = rn[0] #mean
       s = rn[1]+rn[2] # sum, phenotypic variance
       r = rn[1]/(rn[1]+rn[2])#ratio of var components
       
       xyz = numpy.vstack([r,s,m])
                  
       cm = plt.get_cmap("viridis")
       fig = plt.figure()
       ax = fig.add_subplot(111, projection='3d')
       ax.set_xlabel('Variance composition(r)')
       ax.set_ylabel('Phenotypic variance(s)')
       ax.set_zlabel('Midpoint(m)')
       ax.set_xlim3d(0, 1)
       ax.set_ylim3d(0, 0.78)
       ax.set_zlim3d(0, 10)
       #ylim: among = 0.527, within = 0.25
       try:
           density = stats.gaussian_kde(xyz)(xyz)
           idx = density.argsort()
           x, y, z, density = r[idx], s[idx], m[idx], density[idx]
           ax.scatter(x, y, z, c=density, cmap = cm)
       except:
           ax.scatter(r,s,m)
       plt.show()
        
def plot_summary(model_array, variable = 0):
    for i in range(len(model_array)):
        res = model_array[i].results
        val_y = res[variable]
        val_n = res[variable+3]
        plt.scatter(numpy.full(len(val_y),i), val_y, 
                c = sum(val_n)/val_n, cmap = "Blues_r", s = 10, marker="s")
    

n = 10000
outcomes = []
for i in range(5):
    outcomes.append(Run_Program(max_year = n, saving = True, model_name = "sigma"+
                                str(i/2), winter_list = make_climate(
                                        sigma =i/2, n = n), gr_awake=2))
    outcomes[i].run()

plot_3d(outcomes[0].details_list[0])
'''
s0 = Run_Program(max_year = n, saving = True, model_name ="s0", winter_list =
                 make_climate(sigma = 0, n = n), gr_awake = 2)
s0.run()

s1 = Run_Program(max_year = n, saving = True, model_name ="s1", winter_list =
                 make_climate(sigma = 0.5, n = n), gr_awake = 2)
#s1.run()
s2 = Run_Program(max_year = n, saving = True, model_name ="s2", winter_list =
                 make_climate(sigma = 1, n = n), gr_awake = 2)
s2.run()
s3 = Run_Program(max_year = n, saving = True, model_name ="s3", winter_list =
                 make_climate(sigma = 1.5, n = n), gr_awake = 2)
#s3.run()
s4 = Run_Program(max_year = n, saving = True, model_name ="s4", winter_list =
                 make_climate(sigma = 2, n = n), gr_awake = 2)
#s4.run()
s5 = Run_Program(max_year = n, saving = True, model_name ="s5", winter_list =
                 make_climate(sigma = 2.5, n = n), gr_awake = 2)
#s5.run()

#model_list = [s0,s1,s2,s3,s4,s5]
#plot_summary(model_list,0)
'''
#plot parameter space
'''
l = []
for i in range(1000):
    rn = Individual().pars()
    m = rn[0] #mean
    s = rn[1]+rn[2] # sum, phenotypic variance
    r = rn[1]/(rn[1]+rn[2])#ratio of var components
    l.append([m,s,r])
    


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Variance composition(r)')
ax.set_ylabel('Phenotypic variance(s)')
for i in range(len(l)):
    ax.scatter(l[i][1],l[i][2],l[i][0])
ax.set_zlabel('Midpoint(m)')

results = numpy.array(test.details_list)
r = numpy.mean(results, axis=2)
plt.plot(r[:,0])



test.details_list[0]
n=[]
for i in range(len(test.details_list)):
    n.append(numpy.argmax(test.details_list[i][3]))
    
m = [test.details_list[i][0][n[i]] for i in range(len(test.details_list))]

n=[]
for i in range(len(test.details_list)):
    n.append(numpy.argmax(test.details_list[i][4]))
a = [test.details_list[i][1][n[i]] for i in range(len(test.details_list))]


variable =0
for i in range(len(test.details_list)):
    results = test.details_list[i]
    val_y = results[variable]
    val_n = results[variable+3]
    plt.scatter(numpy.full(len(val_y),i), val_y, 
                c = sum(val_n)/val_n, cmap = "Blues_r", s = 10, marker="s")
'''
#and try canalization

