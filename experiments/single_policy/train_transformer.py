from epicf_benchmark.transformer import build_train_parser, train_transformer


if __name__ == "__main__":
    parser = build_train_parser()
    parser.set_defaults(policy_dim=1, epochs=100, model_name="transformer.pt")
    train_transformer(parser.parse_args())
