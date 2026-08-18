"""
Tests the viability of intermediate activation quantization.
Switch between 16-bit and 8-bit quantized for different bandwidths for all the LLMs and measure the average network latency.

worker-A: cpu 10GB
worker-B: cuda 
"""

import os
import subprocess
import time
import re
import csv

MODELS = [
    "models/Llama-3.2-3B-Instruct",
]
BANDWIDTHS = ["3"]

QUANTIZE_FLAGS = ["0", "1"]


def wait_for_container_log(service_name: str, target_string: str, timeout: int=120):
    print(f"Waiting for {service_name} to output: {target_string}...")
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

def run_benchmarks():
    results = []

    for model in MODELS:
        for bw in BANDWIDTHS:
            for quant in QUANTIZE_FLAGS:
                print(f"\nTesting -> Model: {model} | Bandwidth: {bw} Mbps | Quantize: {quant}\n")

                target_log = "Server is running..."

                # Run worker-A
                worker_a_num_layers = 5
                subprocess.run(["./scripts/run_worker.sh", "-m", f"/{model}", "-q", f"{quant}", "-l", "INFO", "-n", f"{worker_a_num_layers}", "worker-A"], check=True)

                wait_for_container_log("worker-A", target_log)
                

                # Run worker-B
                subprocess.run(["./scripts/run_worker.sh", "-m", f"/{model}", "-q", f"{quant}", "-l", "INFO", "worker-B"], check=True)
                wait_for_container_log("worker-B", target_log)
            

                # Trigger client generation
                # Sto docker-compose:
                # command: python distributed_chatbot.py --prompt 'write a poem' --max_new_tokens 100

                print("Triggering token generation...")
                
                env = os.environ.copy()
                env["MODEL_PATH"] = f"/{model}"
                env["LOG_LEVEL"] = "DEBUG"
                env["QUANTIZE"] = str(quant)

                subprocess.run(
                    ["docker", "compose", "up", "-d", "client-A"], 
                    env=env,
                    check=True
                )

                wait_for_container_log("client-A", "Tokens Generated: 100")
                
                
                # Start pumba bandwidth limit
                pumba_cmd = [
                    "docker", "run", "-d", "--rm", "--name", "pumba_netem",
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "ghcr.io/alexei-led/pumba:latest",
                    "netem", "--duration", "1m", "rate", "--rate", f"{bw}mbit",
                    "re2:^decentralized-llm-inference-thesis"
                ]
                subprocess.run(pumba_cmd, check=True)

                wait_for_container_log("client-A", "---------Response---------")

                # Fetch logs
                client_logs = subprocess.run(["docker", "compose", "logs", "client-A"], capture_output=True, text=True).stdout

                tokens_generated = [int(x) for x in re.findall(r"Tokens Generated: (\d+)", client_logs)]
                timestamps = [float(x) for x in re.findall(r"Token Generated at: (\d+\.\d+)", client_logs)]

                max_timestamp = min(timestamps)
                timestamps = [x - max_timestamp for x in timestamps]

                results.append({
                    "Model": model,
                    "Bandwidth_Drop": bw,
                    "Quantize": quant,
                    "Tokens_Generated": tokens_generated,
                    "Timestamps": timestamps
                })

                # Teardown
                subprocess.run(["docker", "rm", "-f", "pumba_netem"], check=False)
                subprocess.run(["docker", "compose", "down", "-v"], check=True)
                time.sleep(5)

    with open("tests/quantization/throughput_jitter.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
                



if __name__ == "__main__":
    run_benchmarks()