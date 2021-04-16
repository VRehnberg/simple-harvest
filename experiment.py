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

def train_agents(env, agents, n_epochs=100, t_max=1000, plot=True):
    '''Train agents for some epochs.'''

    n_agents = env.n_agents
    
    if plot:
        # Axis
        fig, ax = plt.subplots()
        ax.set_xlim([0, n_epochs - 1])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training reward")

        # Colors
        cmap = plt.cm.get_cmap("plasma")
        colors = [cmap(x) for x in np.linspace(0, 1, n_agents)]

        # Theoretical maximum and legend
        cap = env.max_apples
        rate = env.growth_rate
        satmax = (t_max - cap / 2) * rate * cap / 4 + cap / 2
        ax.axhline(satmax, ls="--", c="k", label="SATMax")
        for i_agent, col in enumerate(colors):
            ax.plot(-1, cap / 2, ".", c=col, label=f"Agent{i_agent}")
        if n_agents > 1:
            ax.plot(-1, cap / 2, ".k", label=f"Sum")
        ax.legend()

    for epoch in trange(n_epochs, desc="Epoch"):

        obs = env.reset()
        for i_agent, agent in enumerate(agents):
            obs = env.get_obs(i_agent)
            agent.reset(obs)
            agent.train()

        rewards = np.zeros(n_agents)
        for t in range(t_max):

            actions = [agent.act() for agent in agents]
            all_obs, _, done, info = env.step(*actions)

            for i_agent, agent in enumerate(agents):
                action = actions[i_agent]
                obs = env.get_obs(agent=i_agent)
                reward = env.previous_rewards[i_agent]
                agent.update(action, obs, reward, done)
                rewards[i_agent] += reward

            if done:
                break

        if plot:
            if n_agents > 1:
                ax.plot(epoch, rewards.sum(), 'k.')
            for reward, col in zip(rewards, colors):
                ax.plot(epoch, reward, '.', c=col)
            plt.pause(0.01)
                

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

    # Example run
    qwargs = dict(
        discount=0.95,
        epsilon=0.2,
        epsilon_rate=0.05,
    )
    agent_parameters = [
        # Agent, n, args, kwargs
        (AppleAgent, 0, {}),  # random agents
        (Punisher,   1, {}),
        (QLearner,   2, qwargs),
    ]
    n_agents = sum(n for _, n, _ in agent_parameters)
    max_apples = 20 * n_agents
    env = SimpleHarvest(
        n_agents=n_agents,
        max_apples=max_apples,
        growth_rate=0.05,
    )
    agents = [
        Agent(max_apples, n_agents, **kwargs)
        for Agent, n, kwargs in agent_parameters
        for _ in range(n)
    ]
    train_agents(env, agents)
    run_example(env, agents, render=False)
    embed()
    
    # Visualize apple population logistic growth
    sns.set(font_scale=1.5)
    fig = visualize_growth()
    fig.tight_layout()
    fig.savefig("growth_rate.pdf", bbox_inches="tight")
    sns.set(font_scale=0.6666)

    plt.show()

if __name__=="__main__":
    main()
