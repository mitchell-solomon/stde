import os
import pickle
import string
from pathlib import Path
from typing import Any

import jax
import numpy as np
import shortuuid
from absl import app, flags, logging
from ml_collections import config_dict
from ml_collections.config_flags import config_flags

import stde.equations as eqns
from stde.config import Config
from stde.train import train

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(name="config", default="stde/config.py")
flags.DEFINE_bool("wandb", False, "whether to use wandb")
flags.DEFINE_bool("save", False, "whether to save trained weights")
flags.DEFINE_integer("n_runs", 5, "how many random seeds to run")
flags.DEFINE_bool("get_mem", False, "whether to measure GPU memory usage")

if 'log_dir' in flags.FLAGS:
  default_log_dir = './_logs'
  flags.FLAGS.log_dir = default_log_dir
  Path(default_log_dir).mkdir(parents=True, exist_ok=True)

if 'alsologtostderr' in flags.FLAGS:
  flags.FLAGS.alsologtostderr = True


def main(_: Any) -> None:
  logging.get_absl_handler().use_absl_log_file('absl_logging', FLAGS.log_dir)
  logging.get_absl_handler().setFormatter(
    logging.PythonFormatter("%(message)s")
  )

  cfg: Config = FLAGS.config

  print(cfg)

  # inject random coeffs
  if getattr(eqns, cfg.eqn_cfg.name).random_coeff:
    cfg.eqn_cfg.coeffs = np.random.randn(1, cfg.eqn_cfg.dim)

  uuid = shortuuid.ShortUUID(alphabet=string.ascii_lowercase + string.digits
                            ).random(8)
  if FLAGS.save:
    with cfg.unlocked():
      cfg.uuid = uuid

  l1_losses, l2_losses, iter_per_s_list = [], [], []
  run_data = []
  for i in range(FLAGS.n_runs):
    cfg.rng_seed = i
    np.random.seed(i)
    losses, l1s, l2s, iter_per_s, peak_gpu_mem = train(
      cfg, FLAGS.wandb, i, FLAGS.get_mem
    )
    l1_losses.append(min(l1s))
    l2_losses.append(min(l2s))
    iter_per_s_list.append(iter_per_s)
    run_data.append(
      dict(
        losses=losses,
        l1s=l1s,
        l2s=l2s,
        iter_per_s=iter_per_s,
        peak_gpu_mem=peak_gpu_mem
      )
    )

  l1_losses = np.array(l1_losses)
  l2_losses = np.array(l2_losses)
  iter_per_s = np.array(iter_per_s_list)

  if FLAGS.get_mem:
    result_path = Path("_mem_results/")
  else:
    result_path = Path("_results/")
  result_path.mkdir(parents=True, exist_ok=True)
  result_file_name = "-".join(
    [
      cfg.eqn_cfg.name,
      f"{cfg.eqn_cfg.dim}D",
      f"R{cfg.eqn_cfg.rand_batch_size}",
      f"B{cfg.model_cfg.block_size}",
      uuid,
    ]
  )
  with (result_path / result_file_name).open("w") as f:
    l1_str = f"l1: {l1_losses.mean():.2E}±{l1_losses.std():.2E}\n"
    print(l1_str)
    f.write(l1_str)
    l2_str = f"l2: {l2_losses.mean():.2E}±{l2_losses.std():.2E}\n"
    print(l2_str)
    f.write(l2_str)
    speed_str = f"mean speed: {iter_per_s.mean():.2f}it/s\n"
    print(speed_str)
    f.write(speed_str)
    mem_str = f"peak gpu mem: {peak_gpu_mem:.2f}MBs\n"
    print(mem_str)
    f.write(mem_str)
    f.write(f"\n n_runs: {FLAGS.n_runs}\n")
    f.write("\n---------------CONFIG---------------\n")
    if hasattr(cfg.eqn_cfg, "coeffs"):
      del cfg.eqn_cfg.coeffs
    f.write(str(cfg))

  with (result_path / result_file_name).with_suffix(".pkl").open("wb") as f:
    pickle.dump(run_data, f)


if __name__ == "__main__":
  app.run(main)
