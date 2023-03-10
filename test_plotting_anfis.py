import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from models.anfis import ANFIS
from graph_anfis_functions import plot_anfis_rules

def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ANFIS((1,), 4, layers=[64,64], n_rules=8, membership_type="Triangular").to(device)

    plot_anfis_rules(model)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
