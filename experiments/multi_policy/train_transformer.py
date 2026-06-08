from epicf_benchmark.transformer import build_train_parser, train_transformer


if __name__ == "__main__":
    parser = build_train_parser()
    parser.set_defaults(policy_dim=2, epochs=200, model_name="transformer.pt")
    train_transformer(parser.parse_args())
