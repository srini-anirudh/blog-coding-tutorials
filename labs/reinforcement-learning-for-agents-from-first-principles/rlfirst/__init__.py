"""rlfirst: small, exact experiments for "Reinforcement Learning for Agents From First Principles".

    imitation   behaviour cloning versus DAgger: the O(eps T^2) and O(eps T) regret curves
    shaping     sparse, hand-shaped and potential-based rewards; which cycles a policy can farm
    horizon     P_informative = 1 - p^G - (1 - p)^G, p = q^H, the fifty-percent horizon and recovery
    lock        group-relative RL on a combination lock: empty batches and the fixes that refill them
    credit      trajectory-level versus anchor-state (step-level) advantages on a gridworld
    rollouts    synchronous utilisation E[T] / E[max T] versus an asynchronous queue with a staleness window
"""
