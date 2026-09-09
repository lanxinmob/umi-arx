# OpenVLA 7B on ARX5

The OpenVLA model server and ARX5 control client run in separate Conda
environments. The client reuses the existing UVA camera, teleoperation, delta
pose conversion, recording, and robot bridge.

## Start

Terminal 1, model server:

```bash
conda activate openvla
cd /home/hjadmin/projects/openvla

# This machine's configured proxy currently breaks TLS to Hugging Face.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

python vla-scripts/deploy.py \
  --openvla_path openvla/openvla-7b \
  --attn_implementation sdpa \
  --host 0.0.0.0 \
  --port 8000
```

Terminal 2, robot server:

```bash
conda activate arx5
cd /home/hjadmin/projects/arx5-sdk/python
python communication/zmq_server.py X5_umi can0
```

Terminal 3, camera and ARX5 client:

```bash
conda activate umi-arx
cd /home/hjadmin/projects/umi-arx

python scripts/eval_openvla_arx5.py \
  -o data/experiments/openvla_cup \
  --policy-url http://127.0.0.1:8000/act \
  --task_name 'pick up the cup' \
  --unnorm-key bridge_orig \
  --no_mirror \
  --gripper-mask
```

Use `C` to switch from teleoperation to OpenVLA, `S` to stop an OpenVLA
episode and return to teleoperation, and `Q` to return the arm home and exit.
Keep the hardware emergency stop reachable.

OpenVLA returns one base-frame delta action at a time. The client converts its
XYZ Euler rotation delta to rotvec, applies the same left-multiplied delta pose
conversion as the UVA ARX5 path, maps the absolute gripper value from `[0, 1]`
to `[0, 0.085]` meters, validates step size and episode bounds, and submits one
absolute TCP waypoint.
