from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

import os

def main() -> int:
    dfs = {}
    for file in os.listdir("./runs_graphing"):
        df = pd.read_csv("./runs_graphing/" + file)

        seed = file[file.find("-", 12) + 2:file.rfind("-") - 1]
        model = file[file.rfind("_") + 1:-4]
        dfs[f"{int(seed):03}-{model}"] = df

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    print(dfs["109-dqn"])

    idx = 0
    for idx, seed in enumerate(["009", "019", "043", "109"]):
        specific_keys = [x for x in dfs.keys() if seed in x]
        anfis_key = specific_keys[0]
        dqn_key = specific_keys[1]
        x = int((idx & 2) / 2)
        y = idx % 2
        axs[x, y].set_title(f"Seed: {int(seed)}")
        axs[x, y].plot(dfs[anfis_key]["Step"], dfs[anfis_key]["Value"], label="anfis")
        axs[x, y].plot(dfs[dqn_key]["Step"], dfs[dqn_key]["Value"], label="dqn")
        axs[x, y].set_xlim(0, 100_000)
        axs[x, y].xaxis.set_major_formatter(ticker.FormatStrFormatter("%0.0e"))

    for ax in axs.flat:
        ax.set(xlabel="Training Iterations", ylabel="Median Reward Over 10 Episodes")
        ax.label_outer()

    plt.legend()
    plt.show()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
