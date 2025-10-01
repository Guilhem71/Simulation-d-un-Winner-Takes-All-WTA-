## Simulation d'un Winner Takes All (WTA)

import matplotlib.pyplot as plt
import numpy as np

# parametres du modèle

T = 2000 #durée totale
dt = 0.1 #pas de temps de la stimulation
tan = 10 #temps ajustement du neurone 

stimulus_1 = 2 #intensité du stimulus pour le neurone 1
stimulus_2 = 3 #intensité du stimulus pour le neurone 2

inhib = 5 #force dinhibition du neurone 

sigma_noise = 0.05 # flucttuation aléatoire 

# activation initiale des neurones
a1 = 1.5
a2 = 0.2

# fonction sigmoide pour transformer toute entrée en une valeur entre 0 et 1
def sigmoid(x):
    return 1/(1+np.exp(-x))

# observation évolution de l'activité neuronale dans le temps
a1_trace = np.zeros(T) 
a2_trace = np.zeros(T)

# boucle de simulation du modèle

for t in range (T):
    #bruit aléatoire suivant distribution Gaussienne centré sur 0 et ecart type : sigma.noise 
    noise1 = np.random.normal (0,sigma_noise)
    noise2 = np.random.normal (0,sigma_noise)
    #calcul de la dérivé (-a ramène l'activité vers zero si aucune entrée n'est presente)
    da1= (-a1 + sigmoid(stimulus_1-inhib * a2 + noise1))/tan 
    da2= (-a2 + sigmoid(stimulus_2-inhib * a1 + noise2))/tan
    #mise à jour des activations 
    a1 += da1 * dt
    a2 += da2 * dt
    #stockage pour visualisation
    a1_trace[t] = a1
    a2_trace[t] = a2

# visualiser les résultats

time = np. arange (T)*dt # axe du temps (* dt permet de transformer les indices [0 1 2 3 ...] en instants temporels réels)
plt.plot (time, a1_trace, label = "neurone 1", color = "blue")
plt.plot (time, a2_trace, label = "neurone 2", color = "red")
plt.xlabel ("Temps")
plt.ylabel ("activation")
plt.title("Winner-Take-All par inhibition mutuelle")
plt.legend()
plt.grid ()
plt.show()
