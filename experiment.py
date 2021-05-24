import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, lines, offsetbox
from tqdm import tqdm
from scipy import stats

from simple_harvest import SimpleHarvest
from agents import AppleAgent, Punisher, QLearner
from metrics import GiniRewards, GiniApples, Efficiency, Aggressiveness, GiniTagged, SelfHarm, GiniMetric
from utils import logistic_growth, policy_iteration

PAPER = True
if PAPER:
    import seaborn as sns

    sns.set_context("paper", font_scale=2)
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


def get_markers():
    return itertools.cycle(["o", "s", "X", "D", "v", "^", "<", ">"])


def get_markers_metric():
    return itertools.cycle(["o", "s", "X", "D", "v", "^", "<", ">"])


def train_agents(
        env,
        agents,
        n_trials=1,
        n_epochs=20,
        t_max=1000,
        plot=True,
        metrics=tuple(),
):
    """Train agents for some epochs."""

    n_agents = env.n_agents
    n_metrics = len(metrics)

    # Initialize axes and legends
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
            ax_metric = None

        # Theoretical maximum and legend
        cap = env.max_apples
        rate = env.growth_rate
        max_efficiency = t_max * rate * cap / 4
        ax.axhline(max_efficiency, ls="--", c="k", label="Max")
        colors = []
        markers = get_markers()
        for agent in agents:
            m = next(markers)
            p = ax.plot(-10, max_efficiency / n_agents, f"{m}-", label=repr(agent))
            colors.append(p[0].get_color())
        if n_agents > 1:
            ax.plot(-10, max_efficiency, "P-k", label=f"Sum")
        ax.legend(
            title="Rewards",
            loc='lower center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=3 if n_agents > 2 else 2,
            fancybox=True,
            shadow=True,
        )
        ax.set_xlim([0, n_epochs - 1])
        ax.set_ylim([0, 1.05 * max_efficiency])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.autoscale(enable=False)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training reward")

        # Metrics
        colors_metric = []
        markers_metric = get_markers_metric()
        for metric in metrics:
            m = next(markers_metric)
            p = ax_metric.plot(-10, 0.5, f'{m}-', label=repr(metric))
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
            ax_metric.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        fig.tight_layout()

    # Training loops
    epochs = range(n_epochs)
    all_rewards = np.zeros([n_trials, n_agents, n_epochs])
    all_metrics = np.zeros([n_trials, n_metrics, n_epochs])
    pbar = tqdm(total=n_trials * n_epochs, desc="Train")
    for trial in range(n_trials):
        for agent in agents:
            agent.reinitialize()

        tmp_points = []
        for epoch in epochs:

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

            all_rewards[trial, :, epoch] = rewards
            all_metrics[trial, :, epoch] = [m.value for m in metrics]
            if plot:
                # Add agents progress
                if n_agents > 1:
                    p = ax.plot(epoch, rewards.sum(), '+k')
                    tmp_points.extend(p)
                for reward, col in zip(rewards, colors):
                    p = ax.plot(epoch, reward, '.', c=col)
                    tmp_points.extend(p)

                # Add metrics
                for metric, col in zip(metrics, colors_metric):
                    p = ax_metric.plot(epoch, metric.value, '.', c=col)
                    tmp_points.extend(p)

                # Update plot
                plt.pause(0.01)

            pbar.update(1)

        if plot:
            for p in tmp_points:
                p.remove()

            # Sort rewards
            agent_names = np.array([repr(agent) for agent in agents], dtype=object)
            for agent_type in np.unique(agent_names):
                i_agents = np.flatnonzero(agent_names == agent_type)
                i_sort = i_agents[np.argsort(all_rewards[trial, i_agents, -1])[::-1]]
                all_rewards[trial, i_agents, :] = all_rewards[trial, i_sort, :]

            alpha = 1 / n_trials
            kws = dict(alpha=alpha, ms=4)
            if n_agents > 1:
                ax.plot(epochs, all_rewards[trial, :, :].sum(axis=0), 'P-k', **kws)
            markers = get_markers()
            for i_agent in range(n_agents):
                m = next(markers)
                ax.plot(epochs, all_rewards[trial, i_agent, :], f'{m}-', c=colors[i_agent], **kws)

            # Add metrics
            markers_metric = get_markers_metric
            for i_metric in range(n_metrics):
                col = colors_metric[i_metric]
                m  = next(markers_metric)
                ax_metric.plot(epochs, all_metrics[trial, i_metric, :], f'{m}-', c=col, alpha=alpha)

            plt.pause(0.01)

    if plot and (n_trials > 1):
        # Sort rewards
        for trial in range(n_trials):
            rewards = all_rewards[trial, :, :]
            agent_names = np.array([repr(agent) for agent in agents], dtype=object)
            for agent_type in np.unique(agent_names):
                i_agents = np.flatnonzero(agent_names == agent_type)
                i_sort = i_agents[np.argsort(rewards[i_agents, -1])[::-1]]
                rewards[i_agents, :] = rewards[i_sort, :]
            all_rewards[trial, :, :] = rewards

        def mean_ci(values):
            mean = values.mean(axis=0)
            ci = stats.norm.interval(0.95, loc=values.mean(axis=0), scale=(values).std(axis=0).mean(axis=0))
            ci = np.stack(ci, axis=1)
            return mean, ci

        if n_agents > 1:
            r_sum_mean, r_sum_ci = mean_ci(all_rewards.sum(axis=1, keepdims=True))
        rewards_mean, rewards_ci = mean_ci(all_rewards)
        metrics_mean, metrics_ci = mean_ci(all_metrics)

        def plot_summary(ax, x, y, y_ci, col, markers, n, **kwargs):
            for i in range(n):
                m = next(markers)
                ax.plot(x, y[i, :], c=col[i], marker=m, markeredgecolor="k", **kwargs)
                ax.fill_between(x, y_ci[i, 0, :], y_ci[i, 1, :], fc=col[i], ec=col[i], alpha=0.5, **kwargs)

        kws = dict()
        if n_agents > 1:
            plot_summary(ax, epochs, r_sum_mean, r_sum_ci, ['k'], iter(["P"]), 1, **kws)
        plot_summary(ax, epochs, rewards_mean, rewards_ci, colors, get_markers(), n_agents, **kws)
        plot_summary(ax_metric, epochs, metrics_mean, metrics_ci, colors_metric, get_markers_metric(), n_metrics, **kws)

        plt.pause(0.01)

    pbar.close()


def run_example(env, agents, t_max=1000, render=True):
    """Run a single game with a maximal length."""

    n_agents = env.n_agents

    env.reset()
    for i_agent, agent in enumerate(agents):
        obs = env.get_obs(i_agent)
        agent.reset(obs)
        agent.eval()

    rewards = np.zeros([n_agents, t_max])

    for t in range(t_max):

        if render:
            env.render(mode="human")
            time.sleep(0.5)

        actions = [agent.act() for agent in agents]
        _, _, done, info = env.step(*actions)

        for i_agent, agent in enumerate(agents):
            action = actions[i_agent]
            obs = info[f"obs{i_agent}"]
            reward = info[f"reward{i_agent}"]
            agent.update(action, obs, reward, done)

            rewards[i_agent, t] = reward

        if done:
            break

    avg_rewards = rewards.mean(1)
    for agent, avg_reward in zip(agents, avg_rewards):
        print(f"Reward for {agent}: {avg_reward:.4g}")

    if render:
        fig, ax = plt.subplots()
        t = np.arange(t_max)

        # Theoretical maximum and legend
        cap = env.max_apples
        rate = env.growth_rate
        max_efficiency = rate * cap / 4
        ax.axhline(max_efficiency, ls="--", c="k", label="Max")
        markers = get_markers()
        for i_agent, agent in enumerate(agents):
            m = next(markers)
            ax.plot(t, rewards[i_agent, :], f"{m}", label=repr(agent))
        if n_agents > 1:
            ax.plot(t, rewards.sum(0), "Pk", label=f"Sum")
        ax.legend(
            title="Rewards",
            loc='lower center',
            bbox_to_anchor=(0.5, 0.97),
            ncol=3 if n_agents > 2 else 2,
            fancybox=True,
            shadow=True,
        )
        ax.set_xlim([0, t_max])
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")

        fig.tight_layout()


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
        (QLearner, 2, kwargs),
    ]
    n_agents = sum(n for _, n, _ in agent_parameters)
    growth_rate = 0.15
    max_apples = 20 * n_agents
    n_trials = 3#10
    n_epochs = 20#50
    t_max = 100#1000
    env = SimpleHarvest(
        n_agents=n_agents,
        growth_rate=growth_rate,
        max_apples=max_apples,
    )
    agents = [
        Agent(max_apples, n_agents, **kwargs)
        for Agent, n, kwargs in agent_parameters
        for _ in range(n)
    ]
    metrics = (
        #GiniRewards(n_agents),
        #GiniApples(n_agents),
        #Efficiency(growth_rate, max_apples, t_max),
        #Aggressiveness(n_agents),
        #GiniTagged(n_agents),
        #SelfHarm(n_agents),
    )
    if n_agents == 1:
        metrics = [m for m in metrics if not isinstance(m, GiniMetric)]
    train_agents(env, agents, metrics=metrics, n_trials=n_trials, n_epochs=n_epochs, t_max=t_max)
    run_example(env, agents, t_max=t_max, render=False)

    # Visualize apple population logistic growth
    fig = visualize_growth()
    fig.tight_layout()
    fig.savefig("growth_rate.pdf", bbox_inches="tight")

    # Visualize policy
    max_apples = 20
    growth_rates = np.linspace(0, 1, 500) / (max_apples / 4)
    discounts = np.hstack([np.linspace(0.9, 0.99, 10), np.array([0.995, 0.999])])
    fig, ax = plt.subplots()
    for discount in tqdm(discounts):
        lowest_pick = np.zeros_like(growth_rates, dtype=int)
        highest_wait = np.zeros_like(growth_rates, dtype=int)
        for i, growth_rate in enumerate(growth_rates):
            q_matrix = policy_iteration(growth_rate, max_apples, discount)
            policy = q_matrix.argmax(axis=1)
            lowest_pick[i] = np.min(np.flatnonzero(policy))
            highest_wait[i] = lowest_pick[i] - 1
            assert highest_wait[i] < max_apples
            assert (policy[lowest_pick[i]:] == 1).all()
            assert (policy[:highest_wait[i] + 1] == 0).all()

        # Reduce point density
        strategy_threshold = (highest_wait + lowest_pick) / 2
        sparse_growth = [growth_rates[0]]
        sparse_threshold = [strategy_threshold[0]]
        for i_change in np.flatnonzero(np.diff(strategy_threshold)):
            sparse_threshold.append(strategy_threshold[i_change])
            sparse_threshold.append(strategy_threshold[i_change + 1])
            growth_rate = growth_rates[i_change:i_change + 2].mean()
            sparse_growth.extend(2 * [growth_rate])
        sparse_growth.append(growth_rates[-1])
        sparse_threshold.append(strategy_threshold[-1])

        color = plt.cm.plasma((discount - 0.9) * 10)
        label = r"$\gamma=$" + f"{discount:.4g}"
        marker_on = np.ones_like(sparse_growth, dtype=bool)
        marker_on[-1] = False
        ax.plot(sparse_threshold, sparse_growth, ".-", label=label, color=color, markevery=marker_on)

    # Add legends
    legend = ax.legend(loc='lower right')
    legend_frame = legend.get_frame()
    box1 = offsetbox.TextArea(" Wait ")
    box2 = offsetbox.DrawingArea(10, 20)
    box2.add_artist(lines.Line2D([5, 5], [0, 20], color="k"))
    box3 = offsetbox.TextArea(" Pick ")
    box = offsetbox.HPacker(children=[box1, box2, box3], align="center", pad=legend.borderpad, sep=0)
    abox = offsetbox.AnchoredOffsetbox(
        loc='upper right', child=box, frameon=True, bbox_transform=ax.transAxes, borderpad=legend.borderpad,
    )
    abox.patch.set_boxstyle(legend_frame.get_boxstyle())
    abox.patch.set_facecolor(legend_frame.get_facecolor())
    abox.patch.set_edgecolor(legend_frame.get_edgecolor())
    ax.add_artist(abox)
    ax.set_xlabel("Number of apples, $P$")
    ax.set_ylabel("Growth rate, $r$")

    # Fix axes
    xlim = (-0.5, max_apples + 0.5)
    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(xlim[0], xlim[1] + 1)))
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.xaxis.set_minor_locator(ticker.MaxNLocator(integer=True))
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.set_xlim(xlim)
    ax.autoscale(enable=True, axis='y', tight=True)

    # Save figure
    fig.tight_layout()
    fig.savefig("policy_grid.pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
