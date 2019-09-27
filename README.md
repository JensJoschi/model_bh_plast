This is a simple model of reaction norm evolution along a bet-hedging-plasticity
continuum. Other models on bet-hedging/plasticity either assume two reaction norms
evolving (1 to a predictable cue = plasticity, 1 to white noise; e.g. Tufto 2015),
or an independent contribution of 1 reaction norm and an instability locus 
(e.g. Scheiner2014). This model assumes that there is a single polyphenic reaction 
norm evolving, its shape determines whether we call this strategy bet-hedging
or plasticity (see also  https://doi.org/10.1101/752881 and 10.32942/osf.io/trg34).
The model simulates a multivoltine species that has to diapause before winter onset,
but winter onset is not predictable across years. Depending on the standard 
deviation of winter onset, more plasticity or more bet-hedging is expected to evolve.

Each individual has four properties (b,c,d,e) that determine logistic reaction 
norm shape in response to t (time, e.g. day length). In each time step t, each 
individual decides whether to go to diapause, based on its reaction norm shape. 
If diapausing the individual is transferred to a seed bank (egg bank) of infinite size; 
if not, it will remain and reproduce. When t reaches winter onset,  all remaining 
individuals will be removed and the seed bank replaces the current population.
The next year begins and a new winter onset is drawn from a normal distribution 
with a given µ and sigma. 
A starting population has 100 individuals each from 10 genotypes 
(different bcde combinations), and the simulation runs for s = 10000 years. 
Growth rate is 3 offspring per individual and time step; µ =20, sigma = 0.

The situation is a bit different from standard models on plant germination, 
because the time to start diapausing varies, but all individuals hatch next year.
Still, the seed/egg bank ensures survival under bad and unpredictable conditions.