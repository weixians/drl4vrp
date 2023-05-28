import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.actor import DRL4TSP
from models.critic import StateCritic
from trainer import device, train, validate


def run_tsp(args):
    # Goals from paper:
    # TSP20, 3.97
    # TSP50, 6.08
    # TSP100, 8.44

    from tasks import tsp
    from tasks.tsp import TSPDataset

    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 1  # dummy for compatibility

    train_data = TSPDataset(args.num_nodes, args.train_size, args.seed)
    valid_data = TSPDataset(args.num_nodes, args.valid_size, args.seed + 1)

    update_fn = None

    actor = DRL4TSP(
        STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size, update_fn, tsp.update_mask, args.num_layers, args.dropout
    ).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs["train_data"] = train_data
    kwargs["valid_data"] = valid_data
    kwargs["reward_fn"] = tsp.reward
    kwargs["render_fn"] = tsp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, "actor.pt")
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, "critic.pt")
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = TSPDataset(args.num_nodes, args.train_size, args.seed + 2)

    test_dir = "test"
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, tsp.reward, tsp.render, test_dir, num_plot=5)

    print("Average tour length: ", out)


def run_vrp(args):
    # Goals from paper:
    # VRP10, Capacity 20:  4.84  (Greedy)
    # VRP20, Capacity 30:  6.59  (Greedy)
    # VRP50, Capacity 40:  11.39 (Greedy)
    # VRP100, Capacity 50: 17.23  (Greedy)

    from tasks import vrp
    from tasks.vrp import VehicleRoutingDataset

    # Determines the maximum amount of load for a vehi3**cle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)

    max_load = LOAD_DICT[args.num_nodes]

    train_data = VehicleRoutingDataset(args.train_size, args.num_nodes, max_load, MAX_DEMAND, args.seed)

    valid_data = VehicleRoutingDataset(args.valid_size, args.num_nodes, max_load, MAX_DEMAND, args.seed + 1)

    actor = DRL4TSP(
        STATIC_SIZE,
        DYNAMIC_SIZE,
        args.hidden_size,
        train_data.update_dynamic,
        train_data.update_mask,
        args.num_layers,
        args.dropout,
    ).to(device)

    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    kwargs = vars(args)
    kwargs["train_data"] = train_data
    kwargs["valid_data"] = valid_data
    kwargs["reward_fn"] = vrp.reward
    kwargs["render_fn"] = vrp.render

    if args.checkpoint:
        path = os.path.join(args.checkpoint, "actor.pt")
        actor.load_state_dict(torch.load(path, device))

        path = os.path.join(args.checkpoint, "critic.pt")
        critic.load_state_dict(torch.load(path, device))

    if not args.test:
        train(actor, critic, **kwargs)

    test_data = VehicleRoutingDataset(args.valid_size, args.num_nodes, max_load, MAX_DEMAND, args.seed + 2)

    test_dir = "test"
    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)
    out = validate(test_loader, actor, vrp.reward, vrp.render, test_dir, num_plot=5)

    print("Average tour length: ", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combinatorial Optimization")
    parser.add_argument("--seed", default=12345, type=int)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--task", default="vrp")
    parser.add_argument("--nodes", dest="num_nodes", default=20, type=int)
    parser.add_argument("--actor_lr", default=5e-4, type=float)
    parser.add_argument("--critic_lr", default=5e-4, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden", dest="hidden_size", default=128, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--layers", dest="num_layers", default=1, type=int)
    parser.add_argument("--train-size", default=1000000, type=int)
    parser.add_argument("--valid-size", default=1000, type=int)

    args = parser.parse_args()

    # print('NOTE: SETTTING CHECKPOINT: ')
    # args.checkpoint = os.path.join('vrp', '10', '12_59_47.350165' + os.path.sep)
    # print(args.checkpoint)

    if args.task == "tsp":
        run_tsp(args)
    elif args.task == "vrp":
        run_vrp(args)
    else:
        raise ValueError("Task <%s> not understood" % args.task)
