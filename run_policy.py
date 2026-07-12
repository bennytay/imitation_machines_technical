#!/usr/bin/env python3
"""
Serves a trained ACT policy over a local TCP socket.

This has to be a separate process from the ROS2 side: the policy only runs in
the LeRobot venv (Python 3.12), but the arm is only reachable via rclpy
(Python 3.10) -- see policy_bridge_node.py, which is the ROS2-side client of
this server. The two talk over 127.0.0.1 using a length-prefixed JSON
protocol (no new dependency on the rclpy side).

Run this INSIDE the LeRobot venv (Python 3.12), started BEFORE
policy_bridge_node.py so the socket is already listening when the ROS2 side
tries to connect:

    source ~/venvs/lerobot/bin/activate
    python3 run_policy.py --checkpoint checkpoints/act_final

`--checkpoint` must be a `pretrained_model/`-style directory (config.json +
model.safetensors + the bundled normalization stats/processor pipeline saved
by `lerobot-train`), not a single checkpoint file.
"""
import argparse
import base64
import json
import socket
import struct

import numpy as np
import torch
# lerobot.policies.pretrained references packaging.version but only does
# `import packaging` itself -- which doesn't pull in the `version` submodule
# unless something else already has. Importing it explicitly here avoids an
# AttributeError inside from_pretrained() at checkpoint-load time.
import packaging.version  # noqa: F401

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

STATE_DIM = 6


def parse_args():
    parser = argparse.ArgumentParser(description='Serve a trained ACT policy over a local socket.')
    parser.add_argument('--checkpoint', required=True,
                         help='Path to a pretrained_model/ checkpoint directory')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--device', default='cpu',
                         help='Inference is one sample at a time, so cpu is fine (default).')
    return parser.parse_args()


def recv_exact(conn, n):
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_message(conn):
    header = recv_exact(conn, 4)
    if header is None:
        return None
    (length,) = struct.unpack('>I', header)
    payload = recv_exact(conn, length)
    if payload is None:
        return None
    return json.loads(payload.decode('utf-8'))


def send_message(conn, obj):
    payload = json.dumps(obj).encode('utf-8')
    conn.sendall(struct.pack('>I', len(payload)) + payload)


def observation_to_batch(msg, device):
    """Builds the *unbatched* observation dict the ACT pre-processor expects --
    it adds the batch dimension itself (AddBatchDimensionProcessorStep)."""
    qpos = np.array(msg['qpos'][:STATE_DIM], dtype=np.float32)
    img_bytes = base64.b64decode(msg['image_b64'])
    img = np.frombuffer(img_bytes, dtype=np.uint8).reshape(msg['image_shape'])  # HxWx3 uint8

    state = torch.from_numpy(qpos).to(device)
    # HWC uint8 -> CHW float32 in [0, 1], same convention lerobot uses elsewhere
    # (see lerobot.envs.utils.preprocess_observation) for turning a raw camera
    # frame into what the policy's pre-processor pipeline expects.
    img_t = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).to(device)

    return {
        'observation.state': state,
        'observation.images.cam': img_t,
    }


def main():
    args = parse_args()

    print(f'Loading ACT policy from {args.checkpoint}')
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    # config.device travels with the checkpoint (saved as "mps" if trained on
    # this Mac) and is read directly by parts of the model that create fresh
    # tensors on the fly -- policy.to(device) alone only moves the existing
    # parameters, so config.device has to be updated too or those ops target
    # the wrong device.
    policy.config.device = args.device
    policy.to(args.device)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={'device_processor': {'device': args.device}},
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f'Listening on {args.host}:{args.port} -- waiting for policy_bridge_node to connect...')

    conn, addr = server.accept()
    print(f'Connected: {addr}')

    try:
        with torch.no_grad():
            while True:
                msg = recv_message(conn)
                if msg is None:
                    print('Connection closed.')
                    break

                batch = observation_to_batch(msg, args.device)
                batch = preprocessor(batch)
                action = policy.select_action(batch)
                action = postprocessor(action)

                send_message(conn, {'action': action.squeeze(0).cpu().tolist()})
    finally:
        conn.close()
        server.close()


if __name__ == '__main__':
    main()
