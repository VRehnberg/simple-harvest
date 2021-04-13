import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm

from simple_harvest import SimpleHarvest
from agents import AppleAgent, Punisher, QLearner

# Debugging
from IPython import embed
import pdb

PAPER = True
if PAPER:
    import seaborn as sns
    sns.set_context("paper", font_scale=1.5)
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    sns.set(font='sans-serif')

def visualize_growth():
    log_growth = SimpleHarvest.logistic_growth

    # Parameters
    capacity = 1
    population = np.linspace(0, capacity, 101, dtype=float)
    growth_rate = np.logspace(-0.5, -1.5, 5)

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
        label = f"$r={x_growth_rate:.2g}$"
        ax.plot(population, d_population, label=label)[0]

    ax.set_xlabel("Relative population $P/K$")
    ax.set_ylabel("Relative population growth $\Delta P/K$")
    ax.legend()
    
    return fig

def train_agents(env, agents, n_epochs=10000, t_max=100):
    '''Train agents for some epochs.'''

    n_agents = env.n_agents
    
    obs = env.reset()
    for i_agent, agent in enumerate(agents):
        obs = env.get_obs(i_agent)
        agent.reset(obs)
        agent.train()

    for epoch in trange(n_epochs, desc="Epoch"):
        for t in range(t_max):

            actions = [agent.act() for agent in agents]
            all_obs, _, done, info = env.step(*actions)

            for i_agent, agent in enumerate(agents):
                action = actions[i_agent]
                obs = env.get_obs(agent=i_agent)
                reward = env.previous_rewards[i_agent]
                agent.update(action, obs, reward, done)

            if done:
                print("\rGame lasted", t, "iterations.\n")
                break

def run_example(env, agents, t_max=100, render=True):
    '''Run a single game with a maximal length.'''
    
    n_agents = env.n_agents
    
    obs = env.reset()
    for i_agent, agent in enumerate(agents):
        obs = env.get_obs(i_agent)
        agent.reset(obs)
        agent.eval()

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
        print(f"Reward for {agent}: {avg_reward:.4g}")

def main():

    # Visualize apple population logistic growth
    sns.set(font_scale=1.5)
    fig = visualize_growth()
    fig.tight_layout()
    fig.savefig("growth_rate.pdf", bbox_inches="tight")
    sns.set(font_scale=0.6666)

    # Example run
    agent_parameters = [
        # Agent, n, args, kwargs
        (AppleAgent, 0, {}),  # random agents
        (Punisher,   0, {}),
        (QLearner,   1, {"discount": 0.9, "epsilon": 0.5}),
    ]
    n_agents = sum(n for _, n, _ in agent_parameters)
    max_apples = 10 * n_agents
    env = SimpleHarvest(
        n_agents=n_agents,
        max_apples=max_apples,
        growth_rate=0.01,
    )
    agents = [
        Agent(max_apples, n_agents, **kwargs)
        for Agent, n, kwargs in agent_parameters
        for _ in range(n)
    ]
    train_agents(env, agents)
    run_example(env, agents, render=False)
    embed()
    
    plt.show()

if __name__=="__main__":
    main()
