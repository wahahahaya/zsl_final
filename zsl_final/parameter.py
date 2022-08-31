import argparse


def get_parameters():
    parser = argparse.ArgumentParser()

    # Log
    parser.add_argument("--tensorboard_dir", type=str, default="/HDD-1_data/arlen/zsl_final/log/tensorboard")

    # Parameter
    parser.add_argument("--epochs", type=int, default=30)

    # Data
    parser.add_argument("--train_mode", type=str, default="episode")
    parser.add_argument("--dataset_name", type=str, default='SUN')
    parser.add_argument("--ways", type=int, default=4)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--n_batch", type=int, default=300)
    parser.add_argument("--test_batch", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--train_aug", type=str, default='resize_random_crop')
    parser.add_argument("--test_aug", type=str, default='resize_crop')

    # Test
    parser.add_argument("--test_gamma", type=float, default=1.0)

    # Loss
    parser.add_argument("--lamd1", type=float, default=1.0)
    parser.add_argument("--lamd2", type=float, default=0.05)
    parser.add_argument("--lamd3", type=float, default=0.2)
    parser.add_argument("--lamd4", type=float, default=0.1)

    # Solver
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--steps", type=float, default=10)

    # precision options
    parser.add_argument("--dtype", type=str, default='float32', help="Precision of input, allowable: (float32, float16)")

    return parser.parse_args()
