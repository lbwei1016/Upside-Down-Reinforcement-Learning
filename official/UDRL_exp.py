from UDRL_demo import *


def exp_dropout(dropout: float = 0.1):
    return Config(dropout=dropout)


if __name__ == "__main__":
    config = exp_dropout(0.2)
    ud = UpsideDownAgent(config=config)
    print("Using device", ud.device)

    ud.warm_up()
    print(f"Warm-up complete. Starting training.")
    eval_means, eval_medians = [], []

    while ud.total_steps < Config.max_training_steps:
        results = ud.train_step()
        for r in results:
            if r[0] == "eval.on_policy.mean":
                eval_means.append(r[1])
            if r[0] == "eval.on_policy.median":
                eval_medians.append(r[1])
        ud.msg(f"Iteration {ud.iters} complete\n")
