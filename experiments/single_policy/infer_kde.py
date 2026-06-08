from epicf_benchmark.kde import build_infer_parser, infer_kde


if __name__ == "__main__":
    parser = build_infer_parser()
    parser.set_defaults(policy_dim=1)
    infer_kde(parser.parse_args())
