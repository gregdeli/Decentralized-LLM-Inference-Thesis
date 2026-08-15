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
    "models/Llama-3.1-8B-Instruct",
]
BANDWIDTHS = ["10"]

QUANTIZE_FLAGS = ["0", "1"]


def wait_for_container_log(service_name, target_string, timeout=120):
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
                worker_a_num_layers = 15
                subprocess.run(["./scripts/run_worker.sh", "-m", f"/{model}", "-q", f"{quant}", "-l", "INFO", "-n", f"{worker_a_num_layers}", "worker-A"], check=True)

                wait_for_container_log("worker-A", target_log)
                

                # Run worker-B
                subprocess.run(["./scripts/run_worker.sh", "-m", f"/{model}", "-q", f"{quant}", "-l", "INFO", "worker-B"], check=True)
                wait_for_container_log("worker-B", target_log)
                

                # Start pumba bandwidth limit
                # pumba_cmd = [
                #     "docker", "run", "-d", "--rm", "--name", "pumba_netem",
                #     "-v", "/var/run/docker.sock:/var/run/docker.sock",
                #     "ghcr.io/alexei-led/pumba:latest",
                #     "netem", "--duration", "10m", "rate", "--rate", f"{bw}mbit",
                #     "re2:^decentralized-llm-inference-thesis"
                # ]
                # subprocess.run(pumba_cmd, check=True)

                # Trigger client generation
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

                wait_for_container_log("client-A", "Tokens Generated: 20")
                
                
                # Start pumba bandwidth limit
                pumba_cmd = [
                    "docker", "run", "-d", "--rm", "--name", "pumba_netem",
                    "-v", "/var/run/docker.sock:/var/run/docker.sock",
                    "ghcr.io/alexei-led/pumba:latest",
                    "netem", "--duration", "5s", "rate", "--rate", f"{bw}mbit",
                    "re2:^decentralized-llm-inference-thesis"
                ]
                subprocess.run(pumba_cmd, check=True)

                # continueeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee

                # Fetch logs
                # worker_a_logs = subprocess.run(["docker", "compose", "logs", "worker-A"], capture_output=True, text=True).stdout
                # worker_b_logs = subprocess.run(["docker", "compose", "logs", "worker-B"], capture_output=True, text=True).stdout

                # ser_delays = [float(x) for x in re.findall(r"Serialization Delay: (\d+\.\d+)", worker_a_logs)]
                # grpc_overheads = [float(x) for x in re.findall(r"GRPC Overhead: (\d+\.\d+)", worker_a_logs)]
                # deser_delays = [float(x) for x in re.findall(r"Deserialization Delay: (\d+\.\d+)", worker_b_logs)]

                # prefill_ser = ser_delays[0]
                # prefill_grpc = grpc_overheads[0]
                # prefill_deser = deser_delays[0]

                # total_prefill_latency = prefill_ser + prefill_grpc + prefill_deser

                # ser_delays = ser_delays[1:]
                # grpc_overheads = grpc_overheads[1:]
                # deser_delays = deser_delays[1:]

                # avg_ser = sum(ser_delays) / len(ser_delays) if ser_delays else 0
                # avg_grpc = sum(grpc_overheads) / len(grpc_overheads) if grpc_overheads else 0
                # avg_deser = sum(deser_delays) / len(deser_delays) if deser_delays else 0

                # total_decoding_latency = avg_ser + avg_grpc + avg_deser

                results.append({
                    "Model": model,
                    "Bandwidth": bw,
                    "Quantize": quant,
                    "Prefill_Serialization_Delay": prefill_ser,
                    "Prefill_GRPC_Overhead": prefill_grpc,
                    "Prefill_Deserialization_Delay": prefill_deser,
                    "Total_Prefill_Network_Latency": total_prefill_latency,
                    "Decoding_Avg_Serialization_Delay": avg_ser,
                    "Decoding_Avg_GRPC_Overhead": avg_grpc,
                    "Decoding_Avg_Deserialization_Delay": avg_deser,
                    "Total_Decoding_Avg_Network_Latency": total_decoding_latency
                })

                # Teardown
                subprocess.run(["docker", "rm", "-f", "pumba_netem"], check=False)
                subprocess.run(["docker", "compose", "down", "-v"], check=True)
                time.sleep(5)

    with open("tests/quantization/big_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
                



if __name__ == "__main__":
    run_benchmarks()