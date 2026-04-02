CUDA_VISIBLE_DEVICES=1 python -m src.cli.test_vae \
  --video /mnt/umm/users/pufanyi/workspace/cube/dataset/1_step_frames/00000.mp4 \
  --max_area 184320 \
  --num_frames 81 \
  --output storage/outputs/vae_roundtrip.mp4
