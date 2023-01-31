import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision

from models.modules import ANFIS, DQN

def main() -> int:
    torch.manual_seed(19) # using a great prime number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '/files/',
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
                torchvision.transforms.Resize((41, 41)),
            ])
        ),
        batch_size=128,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            '/files/',
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
                torchvision.transforms.Resize((41, 41)),
            ])
        ),
      batch_size=128,
      shuffle=True,
    )

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    in_shape = tuple(example_data.shape[1:])
    out_shape = 10

    # Roughly similar parameters for the models
    model = ANFIS(in_shape, out_shape, layers=[128, 128], n_rules=16, device=device).to(device)
    # model = DQN(in_shape, out_shape, layers=[64, 64]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=.01)
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for it in range(100):
        for data, target in train_loader:
            output = model(data.to(device))

            optimizer.zero_grad()
            loss = loss_fn(output, target.to(device))
            loss.backward()
            optimizer.step()

        print(f"{loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        n_correct = 0
        for data, target in test_loader:
            target = target.to(device)
            output = model(data.to(device))
            output = F.softmax(output, dim=-1)

            pred = output.argmax(1)

            n_correct += (pred == target).sum().item()

    print(f"{n_correct / len(test_loader.dataset):%}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
