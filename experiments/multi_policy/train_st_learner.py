from epicf_benchmark.st_learner import build_parser, run_st_learner


if __name__ == "__main__":
    parser = build_parser()
    parser.set_defaults(
        data_dir="data_multi_final",
        model_dir="models_multi_final",
        output_dir="output_multi_final",
        train_y="y_train_3.csv",
        train_a="a_train_3.csv",
        train_x="x_train_3.csv",
        policy_dim=2,
        train_units=158 * 3,
        test_units=158 * 3,
        seed=3,
        run_id="3_new",
    )
    run_st_learner(parser.parse_args())
