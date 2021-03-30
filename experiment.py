import time
import numpy as np
import matplotlib.pyplot as plt

from simple_harvest import SimpleHarvest
from agents import AppleAgent, Punisher

def visualize_growth():
    log_growth = SimpleHarvest.logistic_growth

    # Parameters
    population = np.arange(0, 101, dtype=float)
    growth_rate = np.logspace(-0.5, -1.5, 5)
    capacity = 100

    fig, ax = plt.subplots()
    for x_growth_rate in growth_rate:
        # Calculate logistic growth
        new_population = log_growth(
            population.copy(),
            x_growth_rate,
            capacity,
            to_int=False,
        )
        d_population = new_population - population

        # Plot results
        label = f"Growth rate {x_growth_rate:.2g}"
        ax.plot(population, d_population, label=label)[0]

    ax.set_xlabel("Population")
    ax.set_ylabel("Population growth")
    ax.legend()
    
    return fig

def run_example(env, agents, t_max=100, render=True):
    
    n_agents = env.n_agents
    
    obs = env.reset()
    for i_agent, agent in enumerate(agents):
        obs = env.get_obs(i_agent)
        agent.reset(obs)

    total_rewards = np.zeros(n_agents)

    for t in range(t_max):

        if render:
            env.render(mode="human")
            time.sleep(0.5)

        actions = [agent.act() for agent in agents]
        _, _, done, info = env.step(*actions)

        for i_agent, agent in enumerate(agents):
            action = actions[i_agent]
            obs = env.get_obs(agent=i_agent)
            reward = env.previous_rewards[i_agent]
            agent.update(action, obs, reward, done)

        total_rewards += env.previous_rewards
        if done:
            break

    avg_rewards = total_rewards / t
    for agent, avg_reward in zip(agents, avg_rewards):
        print(f"Reward for {agent}: {avg_reward:.4g}.")

def main():

    # Visualize apple population logistic growth
    fig = visualize_growth()

    # Example run
    max_apples = 100
    n_random_agents = 2
    n_punisher_agents = 3
    n_agents = n_random_agents + n_punisher_agents
    env = SimpleHarvest(
        n_agents=n_agents,
        max_apples=max_apples,
        growth_rate=0.1,
    )
    obs_space = env.observation_space
    agents = [
        *[AppleAgent(obs_space) for _ in range(n_random_agents)],
        *[Punisher(obs_space) for _ in range(n_punisher_agents)],
    ]
    run_example(env, agents)
    
    plt.show()

if __name__=="__main__":
    main()
