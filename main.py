import torch
# Computational performance
if torch.backends.cudnn.is_available():
    print("Using cudnn backend")
    torch.backends.cudnn.enabled = True
else:
    print("Using torch without cudnn backend, performance might be slower")

from models.agents import Agent
from config import get_config

def main() -> int:
    args = get_config()

    agent = Agent(
        args.model,
        args.env,
        args.layers,
        args.optimizer,
        args.learning_rate,
        args.loss_fn,
        args.replay_buffer_size,
        args.device,
        args.seed,
        args.n_rules,
        args.writer,
    )

    agent.train(
        args.n_iterations,
        args.train_after,
        args.epsilon_start,
        args.epsilon_end,
        args.epsilon_decay,
        args.update_every,
        args.test_every,
        args.test_eps,
        args.batch_size,
        args.gamma,
        args.gradient_steps,
        args.tau,
        args.grad_norm,
        args.print_updates,
    )

    agent.shutdown()

if __name__ == "__main__":
    raise SystemExit(main())
