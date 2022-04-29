import matplotlib.pyplot as plt

from utils import linear_decay_schedule, linear_growth_schedule

if __name__ == "__main__":
    # Epsilon Schedule
    # values = []
    # schedule = lambda n: linear_decay_schedule(n = n, base = 1, rate = 1e-4, min_val = 1e-3)
    # for i in range(1, 20001):
    #     values.append(schedule(i))

    # plt.plot(range(20000), values)
    # plt.title("Epsilon schedule")
    # plt.xlabel("Epsiode")
    # plt.ylabel("Epsilon")
    # plt.tight_layout()
    # plt.savefig("final/results/epsilon_schedule.png")

    # Beta Schedule
    values = []
    schedule = lambda n: linear_growth_schedule(
        n = n,
        base = 0.6,
        max_val = 1,
        rate = 2e-5
        )
    for i in range(1, 22001):
        values.append(schedule(i))

    plt.plot(range(22000), values)
    plt.title("Beta schedule")
    plt.xlabel("Epsiode")
    plt.ylabel("Beta")
    plt.tight_layout()
    plt.savefig("final/results/beta_schedule.png")