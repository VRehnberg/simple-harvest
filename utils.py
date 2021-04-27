import numpy as np


def polygon_area(x, y):
    """Shoelace formula

    https://stackoverflow.com/a/30408825/15399131
    """
    return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    )


def generalized_gini(income):
    """Generalised Gini coeff from Rafinetti, Siletti and Vernizzi

    https://doi.org/10.1007/s10260-014-0293-4
    """
    if (income == 0).all():
        # No income to all is equality
        return 0.0

    n = len(income)
    # Calculate generalised Lorenz curve
    x = np.linspace(0, 1, n + 1)
    y = np.zeros(n + 1)
    y[1:] = np.cumsum(np.sort(income))
    y_lower = y.copy()
    y_lower[1:-1] = y.min()

    total_area = polygon_area(x, y_lower)
    upper_area = polygon_area(x, y)

    return upper_area / total_area


def regular_gini(income):
    """Regular Gini coefficient"""
    if (income == 0).all():
        # No income to all is equality
        return 0.0

    n = len(income)
    # Calculate generalised Lorenz curve
    x = np.linspace(0, 1, n + 1)
    y = np.zeros(n + 1)
    y[1:] = np.cumsum(np.sort(income))

    total_area = y[-1] / 2
    upper_area = polygon_area(x, y)

    return upper_area / total_area


def logistic_growth(population, growth_rate, capacity, to_int=True):
    """dP / dt = r * P * (1 - P / K)"""
    # Calculate population growth
    d_population = growth_rate * population * (
            1 - population / capacity
    )

    if to_int:
        # Probabilistic rounding to integer
        d_population = int(d_population + np.random.rand())
    else:
        population = population.astype(float)

    # Update population size
    population += d_population
    return population


def policy_iteration(growth_rate, max_apples, discount, tag_cost=0.0, tagged_length=10, n_agents=1):
    """Compute Q-matrix with policy iteration."""
    if n_agents > 1:
        raise NotImplementedError("backwards_induction only implemented for single agent.")

    n_states = max_apples + 1
    n_actions = n_agents + 2

    policy = np.zeros(n_states, dtype=int)
    fixed_policy = np.zeros_like(policy, dtype=bool)

    values = np.zeros(n_states)
    fixed_values = np.zeros_like(values, dtype=bool)

    old_population = np.arange(n_states)
    new_population = logistic_growth(old_population, growth_rate, max_apples, to_int=False)
    transition_probabilities = new_population - old_population

    # Values for P=0 is known
    policy[0] = 0
    fixed_policy[0] = True
    values[0] = 0
    fixed_values[0] = True

    # Policy for P>K/2 is known
    i_middle = max_apples // 2
    policy[i_middle + 1:] = 1
    fixed_policy[i_middle + 1:] = True

    # Check if you can go all out anywhere
    sufficient_growth = (transition_probabilities >= 1.0)
    if sufficient_growth.any():
        i_min = np.argmax(sufficient_growth) + 1
        policy[i_min:] = 1
        fixed_policy[i_min:] = True
        values[i_min:] = 1 / (1 - discount)
        fixed_values[i_min:] = True

    def update_values():
        nonlocal values
        updated_values = fixed_values.copy()

        # Add 0-1 transition values
        zero_mask = (policy == 0)
        one_mask  = (policy == 1)
        is_transition = np.logical_and(zero_mask[:-1], one_mask[1:])
        i_transitions = np.nonzero(is_transition)
        for i in i_transitions:
            p = transition_probabilities[i]
            values[i] = discount * p / (1 - discount)
            updated_values[i] = True
            values[i + 1] = 1 + values[i]
            updated_values[i + 1] = True

        def get_value(state):
            nonlocal values
            nonlocal updated_values
            if updated_values[state]:
                return values[state]

            action = policy[state]

            population = state if action != 1 else state - 1
            p = transition_probabilities[population]

            if action==0:
                value = (discount * p / (1 - discount * (1 - p))) * get_value(state + 1)
            elif action==1:
                value = (discount * (1 - p) / (1 - discount * p)) * get_value(state - 1)

            values[state] = value
            updated_values[state] = True

            return value

        for state in range(n_states):
            get_value(state)
            #TODO values are wrong, see debugger
            raise NotImplementedError

    def get_q_values(values):
        q_values = np.zeros([values.size, 2])
        q_values[1:, 1] += 1
        q_values[1:, :] += discount * (
            np.dot(1 - transition_probabilities[1:], values[1:])
            + np.dot(transition_probabilities[:-1], values[1:])
        )
        #TODO this is wrong
        raise NotImplementedError
        return q_values

    policy_converged = False
    while not policy_converged:

        # Compute values
        update_values()

        # Compute policy
        old_policy = policy.copy()
        policy = get_q_values(values).argmax(axis=1)
        policy_converged = (old_policy == policy).all()

    return get_q_values(values)
