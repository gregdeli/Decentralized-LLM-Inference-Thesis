# Dynamic and Fault-Tolerant Collaborative LLM Inference on Heterogeneous Devices

This repository contains the implementation for an undergraduate thesis on distributed LLM inference. The project provides a fully decentralized framework for executing Large Language Models (LLMs) across a peer-to-peer network of heterogeneous devices (CPUs and GPUs) by distributing the model's transformer layers and the final output layer.

---

## 🎯 Main Idea

The core goal is to enable collaborative LLM inference on edge devices without relying on a central coordinator. The system is dynamic and autonomous; server nodes self-organize, dynamically adjust the LLM layers they host, and form a computational chain that adapts to available hardware resources and network states.

---

## 💡 Motivation & System Strengths

Modern LLMs often exceed the memory and compute capacity of individual, resource-constrained edge devices. This project addresses this by pooling the resources of multiple devices.

* **Resource Distribution:** Enables the execution of large models by spreading the computational workload and memory requirements across multiple computers.
* **Decentralization & Resilience:** Operates without a central orchestrator, eliminating single points of failure. Nodes can join or leave dynamically, and the system autonomously maintains a stable, optimal state.
* **Heterogeneity:** Supports a mix of CPU and GPU nodes within the same inference chain.
* **Privacy:** Keeps inference on-device (or within a trusted local network), avoiding dependence on external cloud APIs.

> **Note:** This system is designed for deployment within a high-bandwidth local network to prevent network latency from reducing inference throughput.

---

## 🏛️ System Architecture

The network consists of client nodes and server nodes that automatically organize into a sequential processing chain (HEAD, intermediate nodes, and TAIL).

### Node Roles

* **Client Node:** * Loads only the initial embedding layer.
  * Initiates inference requests and provides a GUI for a chatbot interface.
  * Responsible for repairing the inference chain if the HEAD server fails.
* **Active Server Node:** * Loads a specific range of transformer layers and/or the final output layer.
  * Runs inference on its assigned layers and transmits intermediate activations to its successor.
  * Autonomously monitors chain balance. If a processing imbalance is detected, it can trigger the Reallocation process.
  * Responsible for repairing the chain if its immediate successor fails.
  * **The TAIL Server:** The final active node in the chain. It holds the output layer, calculates logits, and samples the next token.
* **Backup Server Node:** * If all model layers are already served by the network, new nodes join as backups.
  * They remain on standby to assist with chain repairs if an active node fails.
  * Can perform an "opportunistic takeover" of a weaker active node's layers if the backup has a higher processing rate and sufficient memory.

### Networking & State Management

* **Distributed Hash Table (DHT):** Built using the `hivemind` library, the DHT acts as a decentralized metadata store. Nodes publish their IDs, successors, hosted layers, and performance metrics. If a node fails, its metadata remains temporarily available on the DHT, enabling the network to organize a repair.
* **gRPC Communication:** The Python `grpc` library is utilized for all peer-to-peer remote procedure calls, primarily for transmitting intermediate tensor activations between consecutive nodes in the chain.

---

## ⚙️ Core Operations & Adaptive Allocation

The system utilizes several operations to maintain throughput and fault tolerance:

1. **Initialization & Profiling:** When a server joins, it is profiled to determine its processing rate (measured in transformer layer equivalents per second). The network also calculates an "output layer temporal equivalent" to ensure the TAIL server's processing rate is measured consistently against standard transformer layers. Nodes greedily load as many layers as their memory permits.
2. **Reallocation:** If the inference delay across servers is unequal, an imbalance is detected, triggering a reallocation. Servers adjust their layer assignments proportional to their processing rates. If memory constraints prevent a node from taking its ideal share, it adjusts its effective processing rate and loads its maximum possible layers.
3. **Self-Healing (Repairs):** If an active node fails, its predecessor handles the repair. If the HEAD fails, the initiating Client handles the repair. Backups or re-routing are used to replace the orphaned layers.
4. **Opportunistic Takeover:** A backup node (or a highly performant active node) can autonomously take over the layers of a slower node in the active chain to maximize overall throughput.

---

## 🧠 Memory Management & Optimizations

* **Rotating KV Caches:** Server nodes maintain separate Key-Value (KV) caches for different clients, identified by client addresses. To prevent uncontrolled memory expansion, a global maximum sequence length is defined and divided equally among active clients (e.g., a global limit of 8192 tokens means 4 concurrent clients get 2048 tokens each).
* **8-bit Blockwise Quantization:** Intermediate activations are quantized to 8-bit precision before being transmitted over the network via gRPC, significantly reducing network latency.
* **Execution Pipelining:** The active chain functions as an execution pipeline. Multiple clients can generate text simultaneously, keeping the pipeline full, eliminating idle node time, and maximizing total chain throughput.

---

## 🛠️ Implementation Details

* **Language:** Python
* **Core Framework:** PyTorch (CPU and CUDA support)
* **P2P/Networking:** `hivemind`, `grpc`
* **Models:** Features a custom architecture implementation to allow dynamic layer loading for Llama 3 models (specifically Llama 3.2 1B, Llama 3.2 3B, and Llama 3.1 8B).

---

## 🚀 Future Work

* **Partial Layer Recovery:** During takeovers or repairs, enable the replacement node to load a partial subset of the replaced node's layers (if it lacks memory for all of them), rather than strictly requiring a single node to take all orphaned layers or relying solely on a larger backup.
* **Latency-Aware Operations:** Incorporate network latency metrics into the decision-making process for chain repairs and takeovers (though less critical in high-bandwidth local networks).
* **Post-Training Quantization:** Implement post-training quantization to support the execution of even larger models at acceptable inference speeds.
