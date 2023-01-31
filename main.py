from models.agents import Agent

import argparse

parser = argparse.ArgumentParser(
    prog="Reinforcement Learning for gym-style environments",
    description="Train either an ANFIS or DQN model on a gym-style environment",
)

# Model setup
parser.add_argument("-m", "--model", default="dqn", type=str)
parser.add_argument("-e", "--env", default="CartPole-v1", type=str)
parser.add_argument("-l", "--layers", default=[128, 128], nargs="+", type=int)
parser.add_argument("-o", "--optimizer", default="adam", type=str)
parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
parser.add_argument("-loss", "--loss-fn", default="mse", type=str)
parser.add_argument("-rpb", "--replay-buffer-size", default=10_000)
parser.add_argument("-d", "--device", default="cuda", type=str)
parser.add_argument("-s", "--seed", default=0, type=int)
parser.add_argument("-nr", "--n-rules", default=16, type=int)
parser.add_argument("-w", "--writer", default=True, type=bool)

# Model training
parser.add_argument("-ni", "--n-iterations", default=100_000, type=int)
parser.add_argument("-ta", "--train-after", default=10_000, type=int)
parser.add_argument("-es", "--epsilon-start", default=1.0, type=float)
parser.add_argument("-ee", "--epsilon-end", default=.01, type=float)
parser.add_argument("-ed", "--epsilon-decay", default=.2, type=float)
parser.add_argument("-ue", "--update-every", default=1_000, type=int)
parser.add_argument("-te", "--test-every", default=10_000, type=int)
parser.add_argument("-teps", "--test-eps", default=10, type=int)
parser.add_argument("-bs", "--batch-size", default=128, type=int)
parser.add_argument("-g", "--gamma", default=0.99, type=float)
parser.add_argument("-gs", "--gradient-steps", default=1, type=int)
parser.add_argument("-tau", default=1e-3, type=float)
parser.add_argument("-gn", "--grad-norm", default=10, type=int)
parser.add_argument("-pu", "--print_updates", default=True, type=bool)

def main() -> int:
    args = parser.parse_args()
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
