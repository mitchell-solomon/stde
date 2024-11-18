python -m ipdb main.py \
    --config.eqn_cfg.batch_size 100 \
    --config.eqn_cfg.batch_size_boundary 100 \
    --config.eqn_cfg.enforce_boundary=True \
    --config.eqn_cfg.boundary_weight 0.0 \
    --config.test_cfg.batch_size 200 \
    --config.desc "" \
    --config.test_cfg.eval_every 1000 \
    --config.test_cfg.show_stats=False \
    --config.test_cfg.data_on_gpu=False \
    $@
