import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, lines, offsetbox, colors, cm
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

    fig.tight_layout()

    return fig


def get_markers():
    return itertools.cycle(["o", "s", "X", "D", "v", "^", "<", ">"])


def get_markers_metric():
    return itertools.cycle(["o", "s", "X", "D", "v", "^", "<", ">"])


class TrainPlotter:

    def __init__(self, env, n_epochs, t_max, agents, metrics):
        # Axis
        n_agents = len(agents)
        n_metrics = len(metrics)
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
        colors_reward = []
        markers = get_markers()
        for agent in agents:
            m = next(markers)
            p = ax.plot(-10, max_efficiency / n_agents, f"{m}-", label=repr(agent))
            colors_reward.append(p[0].get_color())
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

        self.agents = agents
        self.metrics = metrics
        self.n_agents = n_agents
        self.n_metrics = n_metrics

        self.fig = fig
        self.ax = ax
        self.ax_metric = ax_metric

        self.tmp_points = []
        self.colors_reward = colors_reward
        self.colors_metric = colors_metric

    def update(self, epoch, rewards):
        if self.n_agents > 1:
            p = self.ax.plot(epoch, rewards.sum(), '+k')
            self.tmp_points.extend(p)
        for reward, col in zip(rewards, self.colors_reward):
            p = self.ax.plot(epoch, reward, '.', c=col)
            self.tmp_points.extend(p)

        # Add metrics
        for metric, col in zip(self.metrics, self.colors_metric):
            p = self.ax_metric.plot(epoch, metric.value, '.', c=col)
            self.tmp_points.extend(p)

    def finalize_trial(self, epochs, rewards, metrics, alpha=1.0):
        for p in self.tmp_points:
            p.remove()
        self.tmp_points = []

        # Sort rewards
        agent_names = np.array([repr(agent) for agent in self.agents], dtype=object)
        for agent_type in np.unique(agent_names):
            i_agents = np.flatnonzero(agent_names == agent_type)
            i_sort = i_agents[np.argsort(rewards[i_agents, -1])[::-1]]
            rewards[i_agents, :] = rewards[i_sort, :]

        kws = dict(alpha=alpha, ms=4)
        if self.n_agents > 1:
            self.ax.plot(epochs, rewards.sum(axis=0), 'P-k', **kws)
        markers = get_markers()
        for i_agent in range(self.n_agents):
            m = next(markers)
            self.ax.plot(epochs, rewards[i_agent, :], f'{m}-', c=self.colors_reward[i_agent], **kws)

        # Add metrics
        markers_metric = get_markers_metric()
        for i_metric in range(self.n_metrics):
            col = self.colors_metric[i_metric]
            m = next(markers_metric)
            self.ax_metric.plot(epochs, metrics[i_metric, :], f'{m}-', c=col, alpha=alpha)

    def summarize(self, epochs, all_rewards, all_metrics):
        # Sort rewards
        n_trials = all_rewards.shape[0]
        for trial in range(n_trials):
            rewards = all_rewards[trial, :, :]
            agent_names = np.array([repr(agent) for agent in self.agents], dtype=object)
            for agent_type in np.unique(agent_names):
                i_agents = np.flatnonzero(agent_names == agent_type)
                i_sort = i_agents[np.argsort(rewards[i_agents, -1])[::-1]]
                rewards[i_agents, :] = rewards[i_sort, :]
            all_rewards[trial, :, :] = rewards

        def mean_ci(values):
            mean = values.mean(axis=0)
            ci = stats.norm.interval(0.68, loc=values.mean(axis=0), scale=values.std(axis=0).mean(axis=0))
            ci = np.stack(ci, axis=1)
            return mean, ci

        if self.n_agents > 1:
            r_sum_mean, r_sum_ci = mean_ci(all_rewards.sum(axis=1, keepdims=True))
        rewards_mean, rewards_ci = mean_ci(all_rewards)
        metrics_mean, metrics_ci = mean_ci(all_metrics)

        def plot_summary(ax, x, y, y_ci, cols, markers, n, **kwargs):
            cols = itertools.cycle(cols)
            for i in range(n):
                c = next(cols)
                m = next(markers)
                ax.plot(x, y[i, :], c=c, marker=m, mec="k", **kwargs)
                ax.fill_between(x, y_ci[i, 0, :], y_ci[i, 1, :], fc=c, ec=c, alpha=0.5, **kwargs)

        kws = dict()
        if self.n_agents > 1:
            plot_summary(self.ax, epochs, r_sum_mean, r_sum_ci, ['k'], iter(["P"]), 1, **kws)
        plot_summary(
            ax=self.ax,
            x=epochs,
            y=rewards_mean,
            y_ci=rewards_ci,
            cols=self.colors_reward,
            markers=get_markers(),
            n=self.n_agents,
            **kws,
        )
        plot_summary(
            ax=self.ax_metric,
            x=epochs,
            y=metrics_mean,
            y_ci=metrics_ci,
            cols=self.colors_metric,
            markers=get_markers_metric(),
            n=self.n_metrics,
            **kws,
        )


class QValuePlotter:

    def __init__(self, env, n_trials, agents):
        n_agents = len(agents)
        max_apples = env.max_apples
        n_combos = 2 ** (n_agents - 1)
        n_actions = env.action_space.n
        q_learners = [agent for agent in agents if isinstance(agent, QLearner)]
        n_q_learners = len(q_learners)
        grid = np.zeros([n_trials, (max_apples // 2) + 1, n_combos, n_actions, n_q_learners])

        self.n_trials = n_trials
        self.n_actions = n_actions
        self.agents = agents
        self.n_agents = n_agents
        self.q_learners = q_learners
        self.grid = grid

    @property
    def grid_3d(self):
        shape = self.grid.shape
        new_shape = (shape[0], shape[1] * shape[3], shape[2] * shape[4])
        grid_3d = np.moveaxis(self.grid, [1, 2, 3, 4], [1, 3, 2, 4]).reshape(*new_shape)
        return grid_3d

    def update(self, trial):
        for i_q_learner, agent in enumerate(self.q_learners):
            q_values = agent.q_values.copy()
            for n_apples in range(self.grid.shape[1]):
                for i_combo, other_actions in enumerate(itertools.product([0, 1], repeat=(self.n_agents - 1))):
                    obs = np.hstack([n_apples, *np.array(other_actions)])
                    state = agent.observe_(obs)
                    self.grid[trial, n_apples, i_combo, :, i_q_learner] = q_values[state, :]

    def plot(self):
        with sns.axes_style("white"):
            fig, axs = plt.subplots(self.n_trials, 1, figsize=(12, self.n_trials * 1.5), constrained_layout=True)

        if self.grid.min() < 0.0:
            norm = colors.TwoSlopeNorm(vmin=self.grid.min(), vcenter=0.0, vmax=self.grid.max())
        else:
            norm = colors.TwoSlopeNorm(vmin=0.0 - 1e-5, vcenter=0.0, vmax=self.grid.max())
        cmap = sns.color_palette("vlag", as_cmap=True)
        grid_3d = self.grid_3d
        yticks, yticklabels = zip(*[
            (i + 0.5, f'({", ".join(combo)})')
            for i, combo in enumerate(itertools.product(["0", "1"], repeat=2))
        ])
        for trial, ax in enumerate(axs):
            # Plot
            grid_2d = grid_3d[trial, :, :]
            ax.pcolormesh(grid_2d.T, ec="w", lw=0.5, cmap=cmap, norm=norm)

            # Modify ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            xticklabels = np.round(ax.get_xticks()).astype(int)
            xticks = xticklabels + 0.5
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
            ax.set_xticklabels(xticklabels)

            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)

            # Axis labels
            ax.set_ylabel(" \n ")
        label_ax = fig.add_subplot(111, frameon=False)
        label_ax.grid(False)
        label_ax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        label_ax.set_ylabel("Agent\nObserved actions")
        label_ax.set_xlabel("Available apples")

        # Colorbar
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs, location="right", fraction=0.07)
        cbar.set_label("Q-value")


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
        train_plotter = TrainPlotter(env, n_epochs, t_max, agents, metrics)
        q_value_plotter = QValuePlotter(env, n_trials, agents)

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
                train_plotter.update(epoch, rewards)
                plt.pause(0.01)

            pbar.update(1)

        if plot:
            train_plotter.finalize_trial(
                epochs,
                all_rewards[trial, :, :],
                all_metrics[trial, :, :],
                alpha=(1 / n_trials),
            )
            q_value_plotter.update(trial)
            plt.pause(0.01)

    if plot:
        train_plotter.summarize(epochs, all_rewards, all_metrics)
        q_value_plotter.plot()
        plt.pause(0.1)

    pbar.close()

    figs = [train_plotter.fig, q_value_plotter.fig] if plot else []
    return figs


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


def visualize_policy():
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
    return fig


def experiment_handler(
    learning_rate=1.0,
    learning_rate_change=0.01,
    discount=0.98,
    epsilon=0.2,
    epsilon_change=0.05,
    growth_rate=0.15,
    n_trials = 10,
    n_epochs = 50,
    t_max = 1000,
    Agents=None,
):
    # Example run
    if Agents is None:
        Agents = {
            AppleAgent: 0,
            Punisher: 0,
            QLearner: 2,
        }

    qlearner_kwargs = dict(
        learning_rate=learning_rate,
        learning_rate_change=learning_rate_change,
        discount=discount,
        epsilon=epsilon,
        epsilon_change=epsilon_change,
    )
    n_agents = sum(Agents.values())
    max_apples = 20 * n_agents
    env = SimpleHarvest(
        n_agents=n_agents,
        growth_rate=growth_rate,
        max_apples=max_apples,
    )
    agents = [
        Agent(max_apples, n_agents, **(qlearner_kwargs if isinstance(Agent, QLearner) else {}))
        for Agent, n in Agents.items()
        for _ in range(n)
    ]
    metrics = (
        GiniRewards(n_agents),
        GiniApples(n_agents),
        Efficiency(growth_rate, max_apples, t_max),
        Aggressiveness(n_agents),
        GiniTagged(n_agents),
        SelfHarm(n_agents),
    )
    if n_agents == 1:
        # Gini doesn't make sense if you have a single agent
        metrics = [m for m in metrics if not isinstance(m, GiniMetric)]
    training_figs = train_agents(env, agents, metrics=metrics, n_trials=n_trials, n_epochs=n_epochs, t_max=t_max)
    run_example(env, agents, t_max=t_max, render=False)

    return training_figs


def parameter_search(loop_kwargs, subdir="param_search"):
    if not os.path.isdir(subdir):
        os.path.mkdir(subdir)
    parameter_names = loop_kwargs.keys()
    parameter_lists = loop_kwargs.values()

    for parameters in itertools.product(*parameter_lists):
        kws = {k : v for k, v in zip(parameter_names, parameters)}
        file_id = "_".join(f"{k}-{v}" for k, v in kws.items())

        train_filename = os.path.join(subdir, "train_" + file_id + ".pdf")
        qval_filename = os.path.join(subdir, "qvalues_" + file_id + ".pdf")
        if os.path.isfile(train_filename) or os.path.isfile(qval_filename):
            continue

        train_fig, qval_fig = experiment_handler(**kws)

        train_fig.savefig(train_filename, bbox_inches="tight")
        qval_fig.savefig(train_filename, bbox_inches="tight")


def main():

    # Run experiment
    parameter_search(dict(
        learning_rate_change=[0.001, 0.01, 0.1],
        discount=[0.96, 0.98, 0.99],
        epsilon=[0.1, 0.2, 0.4],
        epsilon_change=[0.02, 0.05],
        growth_rate=[0.05, 0.15],
    ))

    # Visualize apple population logistic growth
    fig = visualize_growth()
    fig.savefig("growth_rate.pdf", bbox_inches="tight")

    # Visualize policy for single agent
    fig = visualize_policy()
    fig.savefig("policy_grid.pdf", bbox_inches="tight")


    plt.show()


if __name__ == "__main__":
    main()
