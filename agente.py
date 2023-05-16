# video: https://www.youtube.com/watch?v=hlv79rcHws0&ab_channel=MachineLearningwithPhil

import os
import torch as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# PPOMemory es una clase que que se usa en PPO para almacenar las experiencias del agente durante el entrenamiento. Con esta información podemos
# calcular lo que se conoce como "surrogate loss" y en base a esto, actualizar la "policy network" y las "value networks" del agente que es lo que
# hace tomar mejores decisiones en el futuro. Consta de dos redes neuronales, una que se encarga de tomar las decisiones (policy network) y otra que
# se encarga de evaluar las decisiones que toma el agente (value network).
class PPOMemory:
    def __init__(self,batch_size):
        # states almacena los distintos estados del ambiente que el agente experimenta
        self.states = []
        # actions almacena las acciones que el agente toma en cada estado
        self.actions = []
        # probs almacena las probabilidades de las acciones que el agente toma en cada estado, de acuerdo a la "policy network"
        self.probs = []
        # vals almacena los valores de la funcion de valor (fitness function) para cada estado, de acuerdo a los "value networks"
        self.vals = []
        # rewards almacena las recompensas que el agente recibe en cada estado
        self.rewards = []
        # dones almacena flags que indican si el episodio termino o no para cada estado
        self.dones = []

        # batch_size es la cantidad de experiencias que se va a usar para entrenar el agente, o sea para actualizar la "policy network" y las "value networks"
        # durante el entrenamiento. En PPO se toman muestras de mini_batches de experiencias para entrenar el agente que se toman de la memoria (o sea PPOMemory).
        # El tamaño de cada mini_batch es un hiperparametro que se puede ajustar dependiendo de la memoria disponible y de la complejidad del ambiente.
        self.batch_size = batch_size

    # la idea de este método es dividir de forma al azar las experiencias almacenadas en la memoria en batches de experiencias para entrenar el agente.
    def generate_batches(self):
        # en este método se tomarán muestras de mini_batches de experiencias para entrenar el agente que se toman de la memoria.
        n_states = len(self.states)
        # batch_start es una lista de indices (guarda la posicion de los estados). La funcion arange genera una lista de numeros enteros desde 0 hasta n_states
        # con un step de batch_size. Por ejemplo, si batch_size = 5 y n_states = 100, batch_start = [0,5,10,15,...,95]. Esto indica que se van a tomar muestras de
        # 5 experiencias de la memoria para entrenar el agente.
        batch_start = np.arange(0,n_states,self.batch_size)
        # indices es una lista que va desde 0 hasta n_states con un step=1 (ya que no se especifica). Por ejemplo, si n_states = 100, indices = [0,1,2,...,99].
        # Esto nos serviá para tomar muestras de experiencias de la memoria.
        indices = np.arange(n_states,dtype=np.int64)
        # np.random.shuffle(indices) mezcla al azar los indices de la lista indices
        np.random.shuffle(indices)
        # batches es una lista de listas. Cada lista interna contiene los indices de los estados que se van a usar para entrenar el agente. Que se toman
        # al azar de la lista indices. Por ejemplo, si batch_size = 5 y n_states = 100, batch_start = [0,5,10,15,...,95] y indices = [75,5,1,...,25,64],
        # entonces batches = [[75,5,1,..,..],...,[..,..,..,25,64]]. Recordar que indices esta mezclado al azar.
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # devolvemos batches junto con los estados, acciones, rewards, probs, dones y vals que corresponden a los indices de cada batch.
        return np.array(self.states), np.array(self.actions), np.array(self.probs), np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches
    
    def store_memory(self,state,action,prob,val,reward,done):
        # este método almacena las experiencias del agente en la memoria
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.vals.append(val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        # este método limpia la memoria
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

# esta clase que implementa la red neuronal que se encarga de tomar las decisiones del agente (policy network)
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        # n_actions es la cantidad de acciones que el agente puede tomar en el ambiente
        # input_dims es la cantidad de features o parametros que el agente puede observar del ambiente (por ejemplo, la posicion del agente, la velocidad, etc)
        # alpha es el learning rate, o sea la tasa de aprendizaje
        # fc1_dims y fc2_dims son la cantidad de neuronas que tiene la capa 1 y 2 de la red neuronal respectivamente
        # chkpt_dir es el directorio donde se va a guardar el modelo entrenado
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        # aca se define un poco sobre la red neuronal, comienza con un nodo por cada input_dim, luego se conecta con fc1_dims nodos, luego con fc2_dims nodos
        # y finalmente con n_actions nodos. La funcion de activacion que se usa es ReLU, excepto para la ultima capa que se usa Softmax.
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,n_actions),
            nn.Softmax(dim=-1)
        )

        # optimizer es el optimizador que se va a usar para entrenar la red neuronal, tiene algunos hiperparametros que se pueden ajustar
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        # device es el dispositivo donde se va a entrenar la red neuronal, puede nuestra CPU o GPU en caso de que sea CUDA-capable (T.cuda.is_available())
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        # to(device) es un método que mueve la red neuronal a la GPU o CPU
        self.to(self.device)

    def forward(self, state):
        # este método es el que se encarga de pasarle un estado al agente y que este devuelva la probabilidad de cada accion que puede tomar
        # la probabilidad es una distribucion de probabilidad de tipo Categorical
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist
    
    def save_checkpoint(self):
        # este método es el que se encarga de guardar el modelo entrenado
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        # este método es el que se encarga de cargar el modelo entrenado
        self.load_state_dict(T.load(self.checkpoint_file))

# esta clase que implementa la red neuronal que se encarga de estimar el valor de cada estado (value network) 
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        # para esta red neuronal solo se necesita la cantidad de features que tiene cada estado del ambiente y el learning rate
        # no es necesario saber que accion va a tomar el agente, solo se necesita saber el valor de cada estado. Todos los atributos
        # que aparecen aca ya fueron explicados en la clase ActorNetwork

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        self.crtic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims,fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        value = self.crtic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

# por ultimo, esta es la clase que implementa el agente. En este ejemplo tomará como input_dims (o sea la cantidad de features que puede observar del ambiente)
# la posicion de la pelota, la posicion de su paleta y la posicion de la paleta del oponente, la direccion de la pelota y la velocidad de la pelota.
class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lamda=0.95, policy_clip=0.2, batch_size=64, N=2048, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lamda = gae_lamda
        
        # se crea la memoria del agente
        self.memory = PPOMemory(batch_size)
        # se crea la red neuronal que se encarga de tomar las decisiones del agente
        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        # se crea la red neuronal que se encarga de estimar el valor de cada estado
        self.critic = CriticNetwork(input_dims, alpha)

    def remember(self, state, action, prob, val, reward, done):
        self.memory.store_memory(state, action, prob, val, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # se le pasa el estado al agente y este devuelve la distribucion de probabilidad de cada accion
        dist = self.actor(state)
        # se calcula el valor del estado
        value = self.critic(state)
        # se samplea la accion que va a tomar el agente de acuerdo a la distribucion de probabilidad
        action = dist.sample()

        # se calcula la probabilidad de la accion que se sampleo. squeeze() es un método que elimina las dimensiones que tienen tamaño 1
        # por ejemplo, si tenemos x=(1,2,3) y hacemos x.squeeze() nos queda x=(2,3). Sirve para eliminar dimensiones innecesarias
        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value
    
    def learn(self):
        for _ in range(self.n_epochs):
            states_arr, actions_arr, old_probs_arr, vals_arr, rewards_arr, dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            # se calcula el advantage de cada estado. El advantage es la diferencia entre el valor del estado y el valor del estado siguiente
            # el valor del estado siguiente es el valor del estado actual menos el reward que se obtuvo al pasar del estado actual al estado siguiente
            # el valor del estado es el valor que estima la red neuronal que se encarga de estimar el valor de cada estado
            for t in range(len(rewards_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr)-1):
                    a_t += discount*(rewards_arr[k] + self.gamma*values[k+1]*(1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lamda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(states_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(actions_arr[batch]).to(self.actor.device)

                # se le pasa el estado al agente y este devuelve la distribucion de probabilidad de cada accion
                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                # se calcula la probabilidad de la accion que se sampleo
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        # se limpia la memoria del agente para que pueda almacenar nuevos estados luego de todos los epochs
        self.memory.clear_memory()
