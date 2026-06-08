from epicf_benchmark.transformer import build_infer_parser, infer_transformer


if __name__ == "__main__":
    parser = build_infer_parser()
    parser.set_defaults(policy_dim=2, prefix="transformer_cf")
    infer_transformer(parser.parse_args())
