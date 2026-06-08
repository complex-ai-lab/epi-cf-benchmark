from epicf_benchmark.kde import build_train_parser, train_kde


if __name__ == "__main__":
    parser = build_train_parser()
    parser.set_defaults(policy_dim=1)
    train_kde(parser.parse_args())
