"""
worker-A: cpu 4GB
worker-B: cpu 4GB 
worker-C: cuda
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

ALLOCATIONS = ["single", "greedy", "optimal"]

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
    worker_results = []
    client_results = []

    for model in MODELS:
        for allocation in ALLOCATIONS:
            print(f"\nTesting -> Model: {model} | Allocation Strategy: {allocation}\n")

            if allocation == "single":
                env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG")
                subprocess.run(
                    ["docker", "compose", "up", "-d", "worker-C"], 
                    env=env,
                    check=True
                )

                wait_for_container_log("worker-C", "Server is running...")

            elif allocation == "greedy":
                env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG")

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

                subprocess.run(
                    ["docker", "compose", "up", "-d", "worker-C"], 
                    env=env,
                    check=True
                )
                wait_for_container_log("worker-C", "Server is running...")                                

            else:
                # Optimal
                env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG", NUM_LAYERS="3")
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

                env = set_env(MODEL_PATH=model, PROFILE="0", QUANTIZE="1", LOG_LEVEL="DEBUG", NUM_LAYERS="26")
                subprocess.run(
                    ["docker", "compose", "up", "-d", "worker-C"], 
                    env=env,
                    check=True
                )
                wait_for_container_log("worker-C", "Server is running...")  
            

            # Start pumba bandwidth limit
            pumba_cmd = [
                "docker", "run", "-d", "--rm", "--name", "pumba_netem",
                "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "ghcr.io/alexei-led/pumba:latest",
                "netem", "--duration", "10m", "rate", "--rate", f"{BANDWIDTH}mbit",
                "re2:^decentralized-llm-inference-thesis"
            ]
            subprocess.run(pumba_cmd, check=True)

            # Start the client
            env = set_env(
                MODEL_PATH=model,
                PROFILE="0",
                QUANTIZE=QUANTIZE_FLAG,
                LOG_LEVEL="DEBUG"

            )
            # docker-compose: 
            # command: python distributed_chatbot.py --prompt 'write a long thesis on sharks' --max_new_tokens 1000
            print("Triggering text generation...")
            subprocess.run(
                ["docker", "compose", "up", "-d", "client-A"], 
                env=env,
                check=True
            )

            wait_for_container_log("client-A", "Average ITL:")

            # Client A measurements
            client_a_logs = subprocess.run(["docker", "compose", "logs", "client-A"], capture_output=True, text=True).stdout

            ttft = float(re.findall(r"Time To First Token: (\d+\.\d+)", client_a_logs)[0])
            throughput = float(re.findall(r"Throughput: (\d+\.\d+)", client_a_logs)[0])
            avg_itl = float(re.findall(r"Average ITL: (\d+\.\d+)", client_a_logs)[0])

            tokens_generated = [int(x) for x in re.findall(r"Tokens Generated: (\d+)", client_a_logs)]

            timestamps = [float(x) for x in re.findall(r"Token Generated at: (\d+\.\d+)", client_a_logs)]
            min_timestamp = min(timestamps)
            timestamps = [x - min_timestamp for x in timestamps]

            client_results.append({
                "Model": model,
                "Allocation": allocation,
                "Tokens_Generated": tokens_generated,
                "Timestamps": timestamps,
                "TTFT": ttft,
                "Throughput": throughput,
                "Avg_ITL": avg_itl
            })

            if allocation == "single":
                workers = ["worker-C"]
            else:
                workers = ["worker-A", "worker-B", "worker-C"]

            for worker in workers:
                # Worker measurements
                worker_logs = subprocess.run(["docker", "compose", "logs", worker], capture_output=True, text=True).stdout

                layers_match = re.search(r"Loading layers: \((\d+)\, (\d+)\)", worker_logs)
                layers = (int(layers_match.group(1)), int(layers_match.group(2)))
                
                mem_usage = float(re.findall(r"Memory Usage: (\d+\.\d+)", worker_logs)[-1])
                vram_usage = float(re.findall(r"VRAM Usage: (\d+\.\d+)", worker_logs)[-1])

                deser_delays = [float(x) for x in re.findall(r"Deserialization Delay: (\d+\.\d+)", worker_logs)]
                inference_delays = [float(x) for x in re.findall(r"Inference Delay: (\d+\.\d+)", worker_logs)]
                logit_sampling_delays = [float(x) for x in re.findall(r"Logit Sampling Delay: (\d+\.\d+)", worker_logs)] if worker == "worker-C" else [0.0]
                ser_delays = [float(x) for x in re.findall(r"Serialization Delay: (\d+\.\d+)", worker_logs)]
                grpc_overheads = [float(x) for x in re.findall(r"GRPC Overhead: (\d+\.\d+)", worker_logs)]

                avg_deser_delay = sum(deser_delays) / len(deser_delays)
                avg_inference_delay = sum(inference_delays) / len(inference_delays)
                avg_logit_sampling_delay = sum(logit_sampling_delays) / len(logit_sampling_delays)
                avg_ser_delay = sum(ser_delays) / len(ser_delays)
                avg_grpc_overhead = sum(grpc_overheads) / len(grpc_overheads)

                worker_results.append({
                    "Model": model,
                    "Allocation": allocation,
                    "Worker": worker,
                    "Layers": layers,
                    "Memory_Usage": mem_usage,
                    "VRAM_Usage": vram_usage,
                    "Avg_Deserialization_Delay": avg_deser_delay,
                    "Avg_Inference_Delay": avg_inference_delay,
                    "Avg_Logit_Sampling_Delay": avg_logit_sampling_delay,
                    "Avg_Serialization_Delay": avg_ser_delay,
                    "Avg_GRPC_Overhead": avg_grpc_overhead

                })

            # Teardown
            subprocess.run(["docker", "rm", "-f", "pumba_netem"], check=False)
            subprocess.run(["docker", "compose", "down", "-v"], check=True)
            time.sleep(5)

    with open("tests/baseline/client_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=client_results[0].keys())
        writer.writeheader()
        writer.writerows(client_results)

    with open("tests/baseline/worker_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=worker_results[0].keys())
        writer.writeheader()
        writer.writerows(worker_results)
                



if __name__ == "__main__":
    run_benchmarks()