"""
worker-A: cpu 4GB 10 layers
worker-B: cpu 4GB 10 layers
worker-C: cpu 4GB 8 layers + output
worker-D: cpu 4GB
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

BANDWIDTH = "250" # mbit
QUANTIZE_FLAG = "1"

# TAKEOVER = ["1", "0"]


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
        print(f"\nTesting REPAIR With Backup-> Model: {model}\n")

        # Start the server nodes 
        env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="INFO", NUM_LAYERS="10")
        
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

        env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG", NUM_LAYERS="12")
        subprocess.run(
            ["docker", "compose", "up", "-d", "worker-C"], 
            env=env,
            check=True
        )
        wait_for_container_log("worker-C", "Server is running...")

        env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="INFO", NUM_LAYERS="")
        subprocess.run(
            ["docker", "compose", "up", "-d", "worker-D"], 
            env=env,
            check=True
        )
        wait_for_container_log("worker-D", "Server is running...")

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
        subprocess.run(
            ["docker", "compose", "up", "-d", "client-A"], 
            env=env,
            check=True
        )      

        # Stop worker-B to initiate the repair after generating 100 tokens
        wait_for_container_log("client-A", f"Tokens Generated: 100")

        subprocess.run(
            ["docker", "compose", "stop", "worker-B"], 
            check=True
        )

        # Wait for the text generation to end
        wait_for_container_log("client-A", "Average ITL:")

        # Tail server measurements
        worker_c_logs = subprocess.run(["docker", "compose", "logs", "worker-C"], capture_output=True, text=True).stdout

        timestamps = [float(x) for x in re.findall(r"Token Generated at: (\d+\.\d+)", worker_c_logs)]

        client_a_logs = subprocess.run(["docker", "compose", "logs", "client-A"], capture_output=True, text=True).stdout

        token_gen_start_time = float(re.search(r"Start Token Generation at: (\d+\.\d+)", client_a_logs).group(1))

        timestamps.insert(0, token_gen_start_time)

        first_timestamp = timestamps[0]
        timestamps = [x - first_timestamp for x in timestamps]

        tokens_generated = [x for x in range(len(timestamps))]

        results.append({
            "Model": model,
            "Tokens_Generated": tokens_generated,
            "Timestamps": timestamps

        })

        # Teardown
        subprocess.run(["docker", "rm", "-f", "pumba_netem"], check=False)
        subprocess.run(["docker", "compose", "down", "-v"], check=True)

    with open("tests/repair/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
                



if __name__ == "__main__":
    run_benchmarks()