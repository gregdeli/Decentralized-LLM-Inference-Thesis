"""
worker-A: cpu 4GB
worker-B: cpu 4GB 
worker-C: cpu 4GB
worker-D: cuda
"""

import subprocess
import time
import re
import csv
from typing import Dict, Optional
import os

MODELS = [
    "models/Llama-3.2-3B-Instruct",
]

NUM_CLIENTS = [1, 2, 3]

BANDWIDTH = "250" # mbit
QUANTIZE_FLAG = "1"


def wait_for_container_log(service_name, target_string, timeout=300):
    print(f"Waiting for {service_name} to output '{target_string}'")
    start = time.time()

    while time.time() - start < timeout:
        result = subprocess.run(
            ["docker", "compose", "logs", service_name],
            capture_output=True,
            text=True
        )

        if target_string in result.stdout or target_string in result.stderr:
            print(f"{service_name} Ready!")
            return True

        time.sleep(2)

    print(f"Timeout reached while waiting for {service_name}")
    return False

def set_env(MODEL_PATH: str, PROFILE: str, QUANTIZE: str, LOG_LEVEL: str,  NUM_LAYERS: Optional[str] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["MODEL_PATH"] = f"/{MODEL_PATH}"
    env["NUM_LAYERS"] = NUM_LAYERS if NUM_LAYERS else ""
    env["PROFILE"] = PROFILE
    env["QUANTIZE"] = QUANTIZE
    env["LOG_LEVEL"] = LOG_LEVEL

    return env


def run_benchmarks():
    results = []

    for model in MODELS:
        for num_clients in NUM_CLIENTS:
            print(f"\nTesting -> Model: {model} | Num Clients: {num_clients}\n")

            # Start the server nodes with optimal allocation
            env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="INFO", NUM_LAYERS="3")
            
            subprocess.run(
                ["docker", "compose", "up", "-d", "worker-A"], 
                env=env,
                check=True
            )
            wait_for_container_log("worker-A", "Server is running...")

            subprocess.run(
                ["docker", "compose", "up", "-d", "worker-B"], 
                env=env,
                check=True
            )
            wait_for_container_log("worker-B", "Server is running...")

            env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG", NUM_LAYERS="27")
            subprocess.run(
                ["docker", "compose", "up", "-d", "worker-C"], 
                env=env,
                check=True
            )
            wait_for_container_log("worker-C", "Server is running...")

            # env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG", NUM_LAYERS="23")
            # subprocess.run(
            #     ["docker", "compose", "up", "-d", "worker-D"], 
            #     env=env,
            #     check=True
            # )
            # wait_for_container_log("worker-D", "Server is running...")

            # Start pumba bandwidth limit
            pumba_cmd = [
                "docker", "run", "-d", "--rm", "--name", "pumba_netem",
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "ghcr.io/alexei-led/pumba:latest",
                "netem", "--duration", "10m", "rate", "--rate", f"{BANDWIDTH}mbit",
                "re2:^decentralized-llm-inference-thesis"
            ]
            subprocess.run(pumba_cmd, check=True)

            # Start the clients
            env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG")
            if num_clients >= 1:
                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-A"], 
                    env=env,
                    check=True
                )
                last_client = "client-A"

            if num_clients >= 2:                        
                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-B"], 
                    env=env,
                    check=True
                )                      
                last_client = "client-B"

            if num_clients >= 3:
                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-C"], 
                    env=env,
                    check=True
                )      
                last_client = "client-C"

            if num_clients >= 4:
                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-D"], 
                    env=env,
                    check=True
                )      
                last_client = "client-D"

            if num_clients >= 5:                                         
                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-E"], 
                    env=env,
                    check=True
                )                         
                last_client = "client-E"

            if num_clients >= 6:                                         
                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-F"], 
                    env=env,
                    check=True
                )                         
                last_client = "client-F"

            wait_for_container_log(last_client, "Average ITL:")

            # Tail server measurements
            worker_c_logs = subprocess.run(["docker", "compose", "logs", "worker-C"], capture_output=True, text=True).stdout

            timestamps = [float(x) for x in re.findall(r"Token Generated at: (\d+\.\d+)", worker_c_logs)]

            client_a_logs = subprocess.run(["docker", "compose", "logs", "client-A"], capture_output=True, text=True).stdout

            token_gen_start_time = float(re.search(r"Start Token Generation at: (\d+\.\d+)", client_a_logs).group(1))

            timestamps.insert(0, token_gen_start_time)

            min_timestamp = min(timestamps)
            timestamps = [x - min_timestamp for x in timestamps]

            tokens_generated = [x for x in range(len(timestamps))]

            results.append({
                "Model": model,
                "Num Clients": num_clients,
                "Tokens_Generated": tokens_generated,
                "Timestamps": timestamps

            })

            # Teardown
            subprocess.run(["docker", "rm", "-f", "pumba_netem"], check=False)
            subprocess.run(["docker", "compose", "down", "-v"], check=True)
            time.sleep(5)

    with open("tests/multi_client/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
                



if __name__ == "__main__":
    run_benchmarks()