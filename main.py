# esto es para probar si PPO funciona

import gym
import numpy as np
from agente import Agent
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    ax.plot(x, scores, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Scores", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    plt.savefig(filename)

if __name__ == '__main__':
    # este es un juego de prueba para ver si todo funciona correctamente
    env = gym.make('CartPole-v1')
    N = 20
    batch_size = 5
    # n_epochs es el numero de veces que se va a entrenar el agente con los datos que se tienen
    n_epochs = 4
    # este es el learning rate
    alpha = 0.0003
    # todas las acciones posibles ya estan definidas en el env asi como el espacio de observaciones (input_dims)
    print(env.action_space.n, env.observation_space.shape)
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha, n_epochs=n_epochs, input_dims=env.observation_space.shape)
    # numero de juegos que se van a jugar
    n_games = 1

    # se grafican los resultados
    figure_file = 'plots/cartpole.png'

    # esto es para ver si el agente aprende
    best_score = env.reward_range[0]
    score_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, info = env.reset()
        print(observation)
        done = False
        score = 0
        while not done:
            # se elige una accion basandono en la observacion actual (los parametros que se obtienen del juego)
            action, prob, val = agent.choose_action(observation)
            # la diferencia etre terminated y truncated es que terminated es cuando el episodio termina por que el agente perdio o gano, mientras que
            # truncated es cuando el episodio termina por que se llego al limite de pasos o tiempo
            observation_, reward, terminated, trucated, info = env.step(action)
            print(reward)
            # definimos done como terminated o truncated
            done = terminated or trucated
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # informacion para ver como va el entrenamiento
        print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'best score %.1f' % best_score, 'time_steps', n_steps, 'learning_steps', learn_iters)

    # unz vez que termina el entrenamiento se grafican los resultados
    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file) 

