#!/usr/bin/env python
"""
visualizer.py

Created: April 2017
"""

from __future__ import print_function, division
import threading
from multiprocessing import Process
from argparse import ArgumentParser
import json
import os
import shutil

import sys
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "..", "web")
sys.path.insert(0, os.path.abspath(os.path.join(WEB_DIRECTORY, "server")))
from RedisMonitor import RedisMonitor
from WebSocketServer import WebSocketServer
from HTTPRequestHandler import makeHTTPRequestHandler

if sys.version.startswith("3"):
    from http.server import HTTPServer
else:
    from BaseHTTPServer import HTTPServer

args = None
app_thread = None

def euler_fixed_angles_app():
    import redis
    import numpy as np
    import time
    from util import redis_encode, redis_decode
    import rotations
    global args

    KEY_PREFIX = "cs223a::euler_fixed_angles::"
    KEY_SET    = KEY_PREFIX + "set"
    KEY_EULER  = KEY_PREFIX + "zyx_euler_angles"
    KEY_FIXED  = KEY_PREFIX + "xyz_fixed_angles"
    KEY_MATRIX = KEY_PREFIX + "matrix"

    redis_db = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)

    redis_db.mset({
        KEY_SET: "",
        KEY_EULER: "0 0 0",
        KEY_FIXED: "0 0 0",
        KEY_MATRIX: "1 0 0; 0 1 0; 0 0 1"
    })

    while True:
        time.sleep(1)

        try:
            val_set = redis_db.get(KEY_SET)
            if val_set == KEY_EULER:
                euler_angles = redis_decode(redis_db.get(KEY_EULER))
                alpha, beta, gamma = euler_angles[0], euler_angles[1], euler_angles[2]
                R = rotations.zyx_euler_angles_to_mat(alpha, beta, gamma)
            elif val_set == KEY_FIXED:
                fixed_angles = redis_decode(redis_db.get(KEY_FIXED))
                gamma, beta, alpha = fixed_angles[0], fixed_angles[1], fixed_angles[2]
                R = rotations.xyz_fixed_angles_to_mat(gamma, beta, alpha)
            elif val_set == KEY_MATRIX:
                R = redis_decode(redis_db.get(KEY_MATRIX))
                alpha, beta, gamma = rotations.mat_to_zyx_euler_angles(R)
            else:
                continue

            redis_db.mset({
                KEY_SET: "",
                KEY_MATRIX: redis_encode(R),
                KEY_EULER: redis_encode([alpha, beta, gamma]),
                KEY_FIXED: redis_encode([gamma, beta, alpha])
            })

        except Exception as err:
            print(err)
            redis_db.mset({
                KEY_SET: "",
                KEY_EULER: "0 0 0",
                KEY_FIXED: "0 0 0",
                KEY_MATRIX: "1 0 0; 0 1 0; 0 0 1"
            })

def gimbal_lock_app():
    import redis
    import numpy as np
    import time
    from util import redis_encode, redis_decode
    import rotations
    global args

    KEY_PREFIX = "cs223a::gimbal_lock::"
    KEY_SET    = KEY_PREFIX + "set"
    KEY_EULER  = KEY_PREFIX + "zyx_euler_angles"
    KEY_FIXED  = KEY_PREFIX + "xyz_fixed_angles"
    KEY_MATRIX = KEY_PREFIX + "matrix"
    KEY_AXIS   = KEY_PREFIX + "rotation_axis"
    KEY_ANGLE  = KEY_PREFIX + "rotation_angle"
    KEY_EULER_DES = KEY_PREFIX + "rotation_euler"

    redis_db = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)

    R = np.eye(3)
    axis = np.array([1, 0, 0])

    redis_db.mset({
        KEY_SET: "",
        KEY_EULER: "0 0 0",
        KEY_FIXED: "0 0 0",
        KEY_MATRIX: redis_encode(R),
        KEY_AXIS: redis_encode(axis),
        KEY_ANGLE: "0"
    })

    while True:
        time.sleep(1)

        try:
            val_set = redis_db.get(KEY_SET)
            if val_set == KEY_EULER:
                euler_angles = redis_decode(redis_db.get(KEY_EULER))
                alpha, beta, gamma = euler_angles[0], euler_angles[1], euler_angles[2]
                R = rotations.zyx_euler_angles_to_mat(alpha, beta, gamma)
            elif val_set == KEY_FIXED:
                fixed_angles = redis_decode(redis_db.get(KEY_FIXED))
                gamma, beta, alpha = fixed_angles[0], fixed_angles[1], fixed_angles[2]
                R = rotations.xyz_fixed_angles_to_mat(gamma, beta, alpha)
            elif val_set == KEY_MATRIX:
                R = redis_decode(redis_db.get(KEY_MATRIX))
                alpha, beta, gamma = rotations.mat_to_zyx_euler_angles(R)
            elif val_set == KEY_AXIS:
                axis = redis_decode(redis_db.get(KEY_AXIS))
                axis /= np.linalg.norm(axis)
                redis_db.mset({
                    KEY_SET: "",
                    KEY_AXIS: redis_encode(axis),
                })
                continue
            elif val_set == KEY_ANGLE:
                theta = redis_decode(redis_db.get(KEY_ANGLE))[0]
                q = rotations.axis_rotation_to_quat(axis, theta)
                R = rotations.quat_to_mat(q).dot(R)
                R = np.maximum(-1, np.minimum(1, R)) # Ensure R doesn't cause arccos problems
                alpha, beta, gamma = rotations.mat_to_zyx_euler_angles(R)
            else:
                continue

            redis_db.mset({
                KEY_SET: "",
                KEY_MATRIX: redis_encode(R),
                KEY_EULER: redis_encode([alpha, beta, gamma]),
                KEY_FIXED: redis_encode([gamma, beta, alpha]),
                KEY_ANGLE: "0"
            })

        except Exception as err:
            print(err)
            R = np.eye(3)
            axis = np.array([1, 0, 0])
            redis_db.mset({
                KEY_SET: "",
                KEY_EULER: "0 0 0",
                KEY_FIXED: "0 0 0",
                KEY_MATRIX: redis_encode(R),
                KEY_AXIS: redis_encode(axis),
                KEY_ANGLE: "0"
            })

def dh_app():
    import redis
    import numpy as np
    import time
    from util import redis_encode, redis_decode
    import rotations
    import links
    global args

    KEY_PREFIX    = "cs223a::dh::"
    KEY_SET       = KEY_PREFIX + "set"
    KEY_Q         = KEY_PREFIX + "q"
    KEY_T_EE_TO_0 = KEY_PREFIX + "T_ee_to_0"
    KEY_DH_TABLE  = KEY_PREFIX + "dh_table"
    KEY_ROBOT     = KEY_PREFIX + "robot"

    redis_db = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)

    def encode_links(arr_links):
        def encode_var(var):
            return var if var is not None else ""
        encoded_links = ["{0},{1},{2},{3}".format(l.a, l.alpha, encode_var(l.d), encode_var(l.theta)) for l in arr_links]
        encoded_links += [",,,"]
        encoded_links = ";".join(encoded_links)
        return encoded_links

    def decode_links(encoded_links):
        def decode_var(var):
            return float(var) if var != "" else None
        encoded_links = [[decode_var(el) for el in link.split(",")] for link in encoded_links.split(";")]
        arr_links = []
        for l in encoded_links:
            try:
                arr_links.append(links.Link(l[0], l[1], l[2], l[3]))
            except Exception as err:
                break
        if not arr_links:
            raise RuntimeError("Could not parse dh_table.")
        return arr_links

    def encode_robot(arr_links, q):
        import json

        Ts = links.T_all_to_prev(arr_links, q)
        quaternions = [rotations.mat_to_quat(T[:3,:3]).array() for T in Ts]
        positions = [T[:3,3] for T in Ts]
        joint_types = links.prismatic_joints(arr_links)
        json_robot = {
            "links": [{"quat": q.tolist(),
                       "pos" : p.tolist(),
                       "type": t} \
                      for q, p, t in zip(quaternions, positions, joint_types)]
        }
        return json.dumps(json_robot)

    arr_links = [links.Link(0, 0, None, 0)]
    q = np.zeros((1,))
    redis_db.mset({
        KEY_SET: "",
        KEY_Q: redis_encode(q),
        KEY_T_EE_TO_0: "1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1",
        KEY_DH_TABLE: encode_links(arr_links),
        KEY_ROBOT: encode_robot(arr_links, q)
    })

    while True:
        time.sleep(1)

        try:
            val_set = redis_db.get(KEY_SET)
            if val_set == KEY_Q:
                q = redis_decode(redis_db.get(KEY_Q))
                T_ee_to_0 = links.T_i_to_j(arr_links, q, len(arr_links), 0)
            elif val_set == KEY_DH_TABLE:
                arr_links = decode_links(redis_db.get(KEY_DH_TABLE))
                n = len(arr_links)
                if n < q.shape[0]:
                    q = q[:n]
                elif n > q.shape[0]:
                    q_old = q
                    q = np.zeros((n,))
                    q[:q_old.shape[0]] = q_old
                T_ee_to_0 = links.T_i_to_j(arr_links, q, len(arr_links), 0)
            else:
                continue

            redis_db.mset({
                KEY_SET: "",
                KEY_Q: redis_encode(q),
                KEY_T_EE_TO_0: redis_encode(T_ee_to_0),
                KEY_DH_TABLE: encode_links(arr_links),
                KEY_ROBOT: encode_robot(arr_links, q)
            })

        except Exception as err:
            print(err)
            arr_links = [links.Link(0, 0, None, 0)]
            q = np.zeros((1,))
            redis_db.mset({
                KEY_SET: "",
                KEY_Q: redis_encode(q),
                KEY_T_EE_TO_0: "1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1",
                KEY_DH_TABLE: encode_links(arr_links),
                KEY_ROBOT: encode_robot(arr_links, q)
            })

def jacobian_app():
    import redis
    import numpy as np
    import time
    from util import redis_encode, redis_decode
    import rotations
    import links
    import jacobian
    global args

    KEY_PREFIX    = "cs223a::jacobian::"
    KEY_SET       = KEY_PREFIX + "set"
    KEY_Q         = KEY_PREFIX + "q"
    KEY_DH_TABLE  = KEY_PREFIX + "dh_table"
    KEY_EE_OFFSET = KEY_PREFIX + "ee_offset"
    KEY_JV        = KEY_PREFIX + "J_v"
    KEY_JW        = KEY_PREFIX + "J_w"
    KEY_ROBOT     = KEY_PREFIX + "robot"

    redis_db = redis.Redis(host=args.redis_host, port=args.redis_port, db=args.redis_db, decode_responses=True)

    def encode_links(arr_links):
        def encode_var(var):
            return var if var is not None else ""
        encoded_links = ["{0},{1},{2},{3}".format(l.a, l.alpha, encode_var(l.d), encode_var(l.theta)) for l in arr_links]
        encoded_links += [",,,"]
        encoded_links = ";".join(encoded_links)
        return encoded_links

    def decode_links(encoded_links):
        def decode_var(var):
            return float(var) if var != "" else None
        encoded_links = [[decode_var(el) for el in link.split(",")] for link in encoded_links.split(";")]
        arr_links = []
        for l in encoded_links:
            try:
                arr_links.append(links.Link(l[0], l[1], l[2], l[3]))
            except Exception as err:
                break
        if not arr_links:
            raise RuntimeError("Could not parse dh_table.")
        return arr_links

    def encode_robot(arr_links, q, ee_offset=None):
        import json

        if ee_offset is None:
            ee_offset = np.zeros((3,))

        Ts = links.T_all_to_prev(arr_links, q)
        quaternions = [rotations.mat_to_quat(T[:3,:3]).array() for T in Ts]
        positions = [T[:3,3] for T in Ts]
        joint_types = links.prismatic_joints(arr_links)
        json_robot = {
            "links": [{"quat": q.tolist(),
                       "pos" : p.tolist(),
                       "type": t} \
                      for q, p, t in zip(quaternions, positions, joint_types)],
            "ee": {
                "pos": ee_offset.tolist()
            }
        }
        return json.dumps(json_robot)

    arr_links_0 = [
        links.Link(0, 0, 0, None),
        links.Link(0.1, np.pi/2, 0, None),
        links.Link(0, -np.pi/2, None, np.pi/2),
        links.Link(0, np.pi/2, 0.1, None)
    ]
    arr_links = arr_links_0
    q = np.array([0, 0, 0.2, 0])
    ee_offset = np.array([0.2, 0, 0])
    redis_db.mset({
        KEY_SET: "",
        KEY_Q: redis_encode(q),
        KEY_DH_TABLE: encode_links(arr_links),
        KEY_EE_OFFSET: redis_encode(ee_offset),
        KEY_JV: redis_encode(jacobian.linear_jacobian(arr_links, q, pos_in_link=ee_offset)),
        KEY_JW: redis_encode(jacobian.angular_jacobian(arr_links, q)),
        KEY_ROBOT: encode_robot(arr_links, q, ee_offset=ee_offset)
    })

    while True:
        time.sleep(1)

        try:
            val_set = redis_db.get(KEY_SET)
            if val_set == KEY_Q:
                q = redis_decode(redis_db.get(KEY_Q))
            elif val_set == KEY_DH_TABLE:
                arr_links = decode_links(redis_db.get(KEY_DH_TABLE))
                n = len(arr_links)
                if n < q.shape[0]:
                    q = q[:n]
                elif n > q.shape[0]:
                    q_old = q
                    q = np.zeros((n,))
                    q[:q_old.shape[0]] = q_old
            elif val_set == KEY_EE_OFFSET:
                ee_offset = redis_decode(redis_db.get(KEY_EE_OFFSET))
            else:
                continue

            redis_db.mset({
                KEY_SET: "",
                KEY_Q: redis_encode(q),
                KEY_DH_TABLE: encode_links(arr_links),
                KEY_EE_OFFSET: redis_encode(ee_offset),
                KEY_JV: redis_encode(jacobian.linear_jacobian(arr_links, q, pos_in_link=ee_offset)),
                KEY_JW: redis_encode(jacobian.angular_jacobian(arr_links, q)),
                KEY_ROBOT: encode_robot(arr_links, q, ee_offset=ee_offset)
            })

        except Exception as err:
            print(err)
            arr_links = arr_links_0
            q = np.array([0, 0, 0.2, 0])
            ee_offset = np.array([0.2, 0, 0])
            redis_db.mset({
                KEY_SET: "",
                KEY_Q: redis_encode(q),
                KEY_DH_TABLE: encode_links(arr_links),
                KEY_EE_OFFSET: redis_encode(ee_offset),
                KEY_JV: redis_encode(jacobian.linear_jacobian(arr_links, q, pos_in_link=ee_offset)),
                KEY_JW: redis_encode(jacobian.angular_jacobian(arr_links, q)),
                KEY_ROBOT: encode_robot(arr_links, q, ee_offset=ee_offset)
            })

def start_app(app_name, app_target):
    global app_thread
    if app_thread is not None:
        print("Killing %s" % app_thread.name)
        app_thread.terminate()
    print("Starting %s app" % app_name)
    app_thread = Process(target=app_target)
    app_thread.start()

def handle_get_request(request_handler, get_vars, **kwargs):
    """
    HTTPRequestHandler callback:

    Serve content inside WEB_DIRECTORY
    """
    global app_thread
    path_tokens = [token for token in request_handler.path.split("/") if token]

    apps = {
        "euler_fixed_angles": euler_fixed_angles_app,
        "gimbal_lock": gimbal_lock_app,
        "dh": dh_app,
        "jacobian": jacobian_app
    }

    # Default to index.html
    if not path_tokens or ".." in path_tokens:
        request_path = "index.html"
    else:
        request_path = os.path.join(*path_tokens)
        file_ext = os.path.splitext(request_path)
        if len(file_ext) == 2 and file_ext[1] == ".html" and file_ext[0] in apps:
            app_name = file_ext[0]
            if app_thread is not None:
                print("Killing %s" % app_thread.name)
                app_thread.terminate()
            print("Starting %s app" % app_name)
            app_thread = Process(target=apps[app_name])
            app_thread.start()

    request_path = os.path.join(WEB_DIRECTORY, request_path)

    # Check if file exists
    if not os.path.isfile(request_path):
        request_handler.send_error(404, "File not found.")
        return

    # Insert ws_port into redis-web-gui.js
    if request_path == os.path.join(WEB_DIRECTORY, "js", "main.js"):
        with open(request_path) as f:
            html = f.read() % {"ws_port": kwargs["ws_port"]}
        request_handler.wfile.write(html.encode("utf-8"))
        return

    # Otherwise send file directly
    with open(request_path, "rb") as f:
        shutil.copyfileobj(f, request_handler.wfile)

def handle_post_request(request_handler, post_vars, **kwargs):
    """
    HTTPRequestHandler callback:

    Set POST variables as Redis keys
    """

    for key, val_str in post_vars.items():
        if type(val_str[0]) is bytes:
            val_json = json.loads(val_str[0].decode("utf-8"))
        else:
            val_json = json.loads(val_str[0])

        try:
            types = (str, unicode)
        except:
            types = (str,)

        if type(val_json) in types:
            val = val_json
        else:
            val = "; ".join(" ".join(row) for row in val_json)
        print("%s: %s" % (key, val))
        kwargs["redis_db"].set(key, val)


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParser(description=(
        "Monitor Redis keys in the browser."
    ))
    parser.add_argument("-hp", "--http_port", help="HTTP Port (default: 8000)", default=8000, type=int)
    parser.add_argument("-wp", "--ws_port", help="WebSocket port (default: 8001)", default=8001, type=int)
    parser.add_argument("-rh", "--redis_host", help="Redis hostname (default: localhost)", default="localhost")
    parser.add_argument("-rp", "--redis_port", help="Redis port (default: 6379)", default=6379, type=int)
    parser.add_argument("-rd", "--redis_db", help="Redis database number (default: 0)", default=0, type=int)
    parser.add_argument("-r", "--refresh_rate", help="Redis refresh rate in seconds (default: 0.05)", default=0.05, type=float)
    parser.add_argument("--realtime", action="store_true", help="Subscribe to realtime Redis SET pubsub notifications")
    args = parser.parse_args()

    # Create RedisMonitor, HTTPServer, and WebSocketServer
    print("Starting up server...\n")
    redis_monitor = RedisMonitor(host=args.redis_host, port=args.redis_port, db=args.redis_db, refresh_rate=args.refresh_rate, realtime=args.realtime)
    print("Connected to Redis database at %s:%d (db %d)" % (args.redis_host, args.redis_port, args.redis_db))
    get_post_args = {"ws_port": args.ws_port, "redis_db": redis_monitor.redis_db}
    http_server = HTTPServer(("", args.http_port), makeHTTPRequestHandler(handle_get_request, handle_post_request, get_post_args))
    ws_server = WebSocketServer(port=args.ws_port)

    # Start HTTPServer
    http_server_process = Process(target=http_server.serve_forever)
    http_server_process.start()
    print("Started HTTP server on port %d" % (args.http_port))

    # Start WebSocketServer
    ws_server_thread = threading.Thread(target=ws_server.serve_forever, args=(redis_monitor.initialize_client,))
    ws_server_thread.daemon = True
    ws_server_thread.start()
    print("Started WebSocket server on port %d\n" % (args.ws_port))

    # Start RedisMonitor
    print("Server ready. Listening for incoming connections.\n")
    redis_monitor.run_forever(ws_server)

    http_server_process.join()
