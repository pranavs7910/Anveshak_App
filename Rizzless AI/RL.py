"""
Mars Rover Bio-Muscle Arm MDP
7.3.5 - Optimal Value Function via Dynamic Programming (Bellman Recursion)
BONUS  - Monte Carlo Simulation using Optimal Policy
"""

import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────
# STATE SPACE
# ─────────────────────────────────────────────
# State: (fatigue, prev_action)
#   fatigue      : int, 0-9  (≥10 → failure, terminal)
#   prev_action  : 'S' (safe) or 'F' (fast)
# t ranges from 0 (before op 1) to 5 (after op 5, terminal)
# We compute V[t][state] for t = 0..5 using backward DP
# V[5][state] = 0  (no more operations)

ACTIONS = ['S', 'F']  # Safe, Fast
MAX_FATIGUE = 10       # ≥ 10 → failure
FAILURE_PENALTY = 10   # penalty deducted from reward on failure
TIME_SAFE = 5
TIME_FAST = 2

# ─────────────────────────────────────────────
# TRANSITION DYNAMICS
# Returns list of (probability, new_fatigue, failed)
# ─────────────────────────────────────────────
def transitions(fatigue, prev_action, action):
    """
    Compute all (prob, new_fatigue, failed) outcomes.
    failed = True if new_fatigue >= 10.
    """
    outcomes = {}  # new_fatigue -> prob

    if action == 'S':
        # Safe: +1 FU (prob 0.8), +2 FU (prob 0.2)
        for delta, p in [(1, 0.8), (2, 0.2)]:
            f_new = fatigue + delta
            outcomes[f_new] = outcomes.get(f_new, 0) + p

    else:  # Fast action
        # Determine if consecutive fast
        if prev_action == 'F':
            # Consecutive fast: +5 FU (0.6), +7 FU (0.4)
            base_increments = [(5, 0.6), (7, 0.4)]
        else:
            # First fast (prev was safe): +3 FU (0.7), +4 FU (0.3)
            base_increments = [(3, 0.7), (4, 0.3)]

        for delta, p_base in base_increments:
            f_after_base = fatigue + delta

            # Check micro-tear: only if fatigue AFTER base increment >= 8
            if f_after_base >= 8:
                # micro-tear: +4 FU extra (prob 0.2), no extra (prob 0.8)
                for extra, p_tear in [(4, 0.2), (0, 0.8)]:
                    f_new = f_after_base + extra
                    p_total = p_base * p_tear
                    outcomes[f_new] = outcomes.get(f_new, 0) + p_total
            else:
                outcomes[f_after_base] = outcomes.get(f_after_base, 0) + p_base

    # Convert to list of (prob, new_fatigue, failed)
    result = []
    for f_new, prob in outcomes.items():
        failed = f_new >= MAX_FATIGUE
        result.append((prob, f_new, failed))
    return result


# ─────────────────────────────────────────────
# REWARD FUNCTION  R = 10 - Time(a) - Penalty(if f' >= 10)
# ─────────────────────────────────────────────
def reward(action, failed):
    time_cost = TIME_SAFE if action == 'S' else TIME_FAST
    penalty = FAILURE_PENALTY if failed else 0
    return 10 - time_cost - penalty


# ─────────────────────────────────────────────
# 7.3.5 — DYNAMIC PROGRAMMING (Backward Induction)
# t=0: before op 1 ... t=4: before op 5 (last op)
# V[5] = 0 (terminal, no more ops)
# ─────────────────────────────────────────────
def compute_optimal_value_function():
    # V[t][(fatigue, prev)] = optimal expected total reward from step t onward
    # policy[t][(fatigue, prev)] = optimal action
    V = [{} for _ in range(6)]
    policy = [{} for _ in range(6)]

    # Terminal: V[5] = 0 for all states
    for f in range(MAX_FATIGUE):
        for prev in ['S', 'F']:
            V[5][(f, prev)] = 0.0

    # Backward recursion: t = 4 down to 0
    for t in range(4, -1, -1):
        for f in range(MAX_FATIGUE):
            for prev in ['S', 'F']:
                best_val = -float('inf')
                best_act = None

                for action in ACTIONS:
                    q_val = 0.0
                    for prob, f_new, failed in transitions(f, prev, action):
                        r = reward(action, failed)
                        if failed:
                            # Episode terminates — no future value
                            future = 0.0
                        else:
                            # next prev_action is current action
                            next_prev = action
                            next_f = min(f_new, MAX_FATIGUE - 1)  # cap for indexing
                            future = V[t+1].get((next_f, next_prev), 0.0)
                        q_val += prob * (r + future)

                    if q_val > best_val:
                        best_val = q_val
                        best_act = action

                V[t][(f, prev)] = best_val
                policy[t][(f, prev)] = best_act

    return V, policy


# ─────────────────────────────────────────────
# BONUS — MONTE CARLO SIMULATION (1000 missions)
# ─────────────────────────────────────────────
def simulate_missions(policy, n_missions=1000, seed=42):
    rng = np.random.default_rng(seed)
    total_rewards = []
    final_fatigues = []
    failures = 0

    for _ in range(n_missions):
        fatigue = 0
        prev = 'S'  # initial: treat as previous was safe
        total_reward = 0
        failed = False

        for t in range(5):  # 5 operations
            action = policy[t].get((fatigue, prev), 'S')

            # Sample outcome
            trans = transitions(fatigue, prev, action)
            probs = [p for p, _, _ in trans]
            idx = rng.choice(len(trans), p=probs)
            _, f_new, is_fail = trans[idx]

            r = reward(action, is_fail)
            total_reward += r
            fatigue = f_new

            if is_fail:
                failed = True
                break
            prev = action

        total_rewards.append(total_reward)
        final_fatigues.append(min(fatigue, MAX_FATIGUE))
        if failed:
            failures += 1

    return {
        'prob_failure': failures / n_missions,
        'avg_total_reward': np.mean(total_rewards),
        'avg_final_fatigue': np.mean(final_fatigues),
        'std_total_reward': np.std(total_rewards),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  MARS ROVER MDP — OPTIMAL VALUE FUNCTION (DP)")
    print("=" * 60)

    V, policy = compute_optimal_value_function()

    # Print V[t] for key states
    print("\n Optimal Values V[t](fatigue, prev) — Selected States\n")
    print(f"{'t':>3} | {'(f, prev)':>10} | {'V[t]':>10} | {'Action':>8}")
    print("-" * 40)

    display_states = [(0,'S'),(0,'F'),(2,'S'),(2,'F'),(4,'S'),(4,'F'),(6,'S'),(6,'F'),(7,'S'),(9,'S')]
    for t in range(5):
        for state in display_states:
            if state in V[t]:
                print(f"{t:>3} | {str(state):>10} | {V[t][state]:>10.4f} | {policy[t][state]:>8}")
        print()

    # Specifically print V1(4, S) as asked in prior question
    print(f"\n V[1](f=4, prev=S) = {V[1].get((4,'S'), 'N/A'):.4f}")
    print(f"   Optimal action    = {policy[1].get((4,'S'), 'N/A')}")

    print(f"\n V[0](f=0, prev=S) = {V[0].get((0,'S'), 'N/A'):.4f}  ← start of mission")
    print(f"   Optimal action    = {policy[0].get((0,'S'), 'N/A')}")

    # Print full policy table
    print("\n" + "=" * 60)
    print("  OPTIMAL POLICY π*(t, f, prev)")
    print("=" * 60)
    print(f"{'t':>3} | {'fatigue':>7} | {'prev=S':>8} | {'prev=F':>8}")
    print("-" * 35)
    for t in range(5):
        for f in range(10):
            a_s = policy[t].get((f,'S'), '-')
            a_f = policy[t].get((f,'F'), '-')
            if a_s != a_f or f < 8:
                print(f"{t:>3} | {f:>7} | {a_s:>8} | {a_f:>8}")

    # BONUS: Simulation
    print("\n" + "=" * 60)
    print("  BONUS — MONTE CARLO SIMULATION (1000 missions)")
    print("=" * 60)

    stats = simulate_missions(policy)
    print(f"\n  Probability of actuator failure : {stats['prob_failure']:.3f}  ({stats['prob_failure']*100:.1f}%)")
    print(f"   Average total reward            : {stats['avg_total_reward']:.4f}  ± {stats['std_total_reward']:.4f}")
    print(f"   Average final fatigue           : {stats['avg_final_fatigue']:.4f} FU")
    print()