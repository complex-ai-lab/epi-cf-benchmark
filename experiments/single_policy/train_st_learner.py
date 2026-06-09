from epicf_benchmark.st_learner import build_parser, run_st_learner


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(
        data_dir="data_single_final",
        model_dir="models_single_final",
        output_dir="output_single_final",
        policy_dim=1,
        train_units=158,
        test_units=158,
        seed=1,
        run_id="1_new",
    )
    run_st_learner(parser.parse_args())
