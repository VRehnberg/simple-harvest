import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from matplotlib.patches import Patch

from simple_harvest import SimpleHarvest
from agents import AppleAgent, Punisher, QLearner
from metrics import GiniRewards, GiniApples
from utils import logistic_growth, policy_iteration

PAPER = True
if PAPER:
    import seaborn as sns

    sns.set_context("paper", font_scale=1.5)
    sns.set_style("darkgrid")
    sns.set_palette("deep")
    sns.set(font='sans-serif')


def visualize_growth():

    # Parameters
    capacity = 1
    population = np.linspace(0, capacity, 101, dtype=float)
    growth_rate = np.logspace(-0.5, -1.5, 5)

    fig, ax = plt.subplots()
    for x_growth_rate in growth_rate:
        # Calculate logistic growth
        new_population = logistic_growth(
            population.copy(),
            x_growth_rate,
            capacity,
            to_int=False,
        )
        d_population = new_population - population

        # Plot results
        label = f"$r={x_growth_rate:.2g}$"
        ax.plot(population, d_population, label=label)

    ax.set_xlabel(r"Relative population $P/K$")
    ax.set_ylabel(r"Relative population growth $\Delta P/K$")
    ax.legend()

    return fig


def train_agents(
        env,
        agents,
        n_epochs=100,
        t_max=1000,
        plot=True,
        metrics=tuple(),
):
    """Train agents for some epochs."""

    n_agents = env.n_agents

    if plot:
        # Axis
        if metrics:
            fig, (ax, ax_metric) = plt.subplots(1, 2, figsize=(12, 4.8))
            ax_metric.set_xlim([0, n_epochs - 1])
            ax_metric.set_ylim([-0.05, 1.05])
            ax_metric.set_xlabel("Epoch")
            ax_metric.set_ylabel("Metric value")
        else:
            fig, ax = plt.subplots()
        ax.set_xlim([0, n_epochs - 1])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training reward")

        # Theoretical maximum and legend
        cap = env.max_apples
        rate = env.growth_rate
        max_efficiency = (t_max - cap / 2) * rate * cap / 4 + cap / 2
        ax.axhline(max_efficiency, ls="--", c="k", label="Max")
        colors = []
        for agent in agents:
            p = ax.plot(-10, cap / 2, ".", label=repr(agent))
            colors.append(p[0].get_color())
        if n_agents > 1:
            ax.plot(-10, cap / 2, ".k", label=f"Sum")
        ax.legend(
            title="Rewards",
            loc='lower center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=3,
            fancybox=True,
            shadow=True,
        )

        # Metrics
        colors_metric = []
        for metric in metrics:
            p = ax_metric.plot(-10, 0.5, '.', label=repr(metric))
            colors_metric.append(p[0].get_color())

        if ax_metric:
            ax_metric.legend(
                title="Metrics",
                loc='lower center',
                bbox_to_anchor=(0.5, 0.97),
                ncol=3,
                fancybox=True,
                shadow=True,
            )

        fig.tight_layout()

    # Training loop
    for epoch in trange(n_epochs, desc="Train"):

        # Reset
        env.reset()
        for i_agent, agent in enumerate(agents):
            obs = env.get_obs(i_agent)
            agent.reset(obs)
            agent.train()

        for metric in metrics:
            metric.reset()

        rewards = np.zeros(n_agents)
        for t in range(t_max):

            actions = [agent.act() for agent in agents]
            all_obs, _, done, info = env.step(*actions)

            # Update agents
            for i_agent, agent in enumerate(agents):
                action = actions[i_agent]
                obs = env.get_obs(agent=i_agent)
                reward = env.previous_rewards[i_agent]
                agent.update(action, obs, reward, done)
                rewards[i_agent] += reward

            # Update metrics
            for metric in metrics:
                metric.update(info)

            if done:
                break

        if plot:
            # Add agents progress
            if n_agents > 1:
                ax.plot(epoch, rewards.sum(), 'k.')
            for reward, col in zip(rewards, colors):
                ax.plot(epoch, reward, '.', c=col)

            # Add metrics
            for metric, col in zip(metrics, colors_metric):
                ax_metric.plot(epoch, metric.value, '.', c=col)

            # Update plot
            plt.pause(0.01)


def run_example(env, agents, t_max=100, render=True):
    """Run a single game with a maximal length."""

    n_agents = env.n_agents

    env.reset()
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
    kwargs = dict(
        learning_rate=1.0,
        learning_rate_change=0.01,
        discount=0.98,
        epsilon=0.2,
        epsilon_change=0.05,
    )
    agent_parameters = [
        # Agent, n, args, kwargs
        (AppleAgent, 0, {}),  # random agents
        (Punisher, 0, {}),
        (QLearner, 3, kwargs),
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
    metrics = (GiniRewards(n_agents), GiniApples(n_agents))
    train_agents(env, agents, metrics=metrics)
    run_example(env, agents, render=False)

    # Visualize apple population logistic growth
    sns.set(font_scale=1.5)
    fig = visualize_growth()
    fig.tight_layout()
    fig.savefig("growth_rate.pdf", bbox_inches="tight")
    sns.set(font_scale=0.6666)

    # Visualize policy
    max_apples = 20
    population = np.linspace(0, max_apples, max_apples + 1)
    growth_rates = np.linspace(0.05, 1, 20) / (max_apples / 4)
    discount = 0.98
    policies = np.zeros([growth_rates.size, population.size])
    for i, growth_rate in enumerate(growth_rates):
        q_matrix = policy_iteration(growth_rate, max_apples, discount)
        policy = q_matrix.argmax(axis=1)
        policies[i, :] = policy

    X, Y = np.meshgrid(population, growth_rates)

    sns.set(font_scale=1.5)
    fig, ax = plt.subplots()
    image = ax.pcolormesh(X, Y, policies, cmap="bwr", edgecolors="w", shading="nearest", lw=0.5)
    legend_entries = zip(*[
        (Patch(facecolor=image.cmap(float(a)), edgecolor='w'), env.action_meanings[a])
        for a in [0, 1]
    ])
    ax.legend(*legend_entries)
    ax.set_xlabel("Number of apples, $P$")
    ax.set_ylabel("Growth rate, $r$")
    fig.tight_layout()
    fig.savefig("policy_grid.pdf", bbox_inches="tight")
    sns.set(font_scale=0.6666)


    plt.show()


if __name__ == "__main__":
    main()
