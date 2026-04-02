CUDA_VISIBLE_DEVICES=1 python -m src.cli.infer_i2v \
  --use_ema \
  --checkpoint storage/checkpoints/cube_test/checkpoint-7000 \
  --image /mnt/umm/users/pufanyi/workspace/cube/dataset/1_step_frames/00000.jpg \
  --prompt "Follow the image notation and apply the moves to the Rubik's cube." \
  --max_area 184320 \
  --num_frames 81 \
  --output storage/outputs/cube_7000_ema.mp4 \
  --seed 42