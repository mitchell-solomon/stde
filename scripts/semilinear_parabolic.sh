python main.py \
    --config.eqn_cfg.unbiased=False \
    --config.eqn_cfg.discretize_time=False \
    --config.eqn_cfg.batch_size 2000 \
    --config.eqn_cfg.batch_size_boundary 100 \
    --config.eqn_cfg.n_traj 100 \
    --config.eqn_cfg.n_t 20 \
    --config.eqn_cfg.T 0.3 \
    --config.eqn_cfg.boundary_weight 20.0 \
    --config.eqn_cfg.boundary_g_weight 0.05 \
    --config.model_cfg.width 1024 \
    --config.gd_cfg.lr_decay exponential \
    --config.eqn_cfg.enforce_boundary=False \
    --config.desc "" \
    --config.test_cfg.eval_every 100 \
    --config.test_cfg.show_stats=False \
    $@
