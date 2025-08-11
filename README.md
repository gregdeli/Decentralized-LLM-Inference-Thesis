# A Decentralized and Adaptive Framework for Distributed LLM Inference

This repository contains the work for the undergraduate thesis: **A Decentralized and Adaptive Framework for Distributed LLM Inference on Heterogeneous Edge Devices**.

---

## 🎯 Main Idea
The core goal of this project is to create a fully decentralized framework for running collaborative, model-distributed LLM inference on a peer-to-peer network of heterogeneous edge devices.

Instead of relying on a central coordinator, the framework uses a self-organizing network of server nodes. These servers dynamically adjust which LLM layers they host, forming a computational chain that automatically adapts to changes in available resources and network performance.

---

## 💡 Motivation
Modern LLMs are too large to run on single, resource-constrained edge devices, creating a dependency on remote cloud servers. This project aims to solve that problem by enabling collaborative inference directly on the edge.  
This approach offers several advantages:

- **Privacy & Availability:** Keeps data on-device and ensures the system works even without internet access.
- **Efficiency:** Maximizes the use of available peer-to-peer resources.
- **Resilience:** Avoids the single point of failure inherent in centrally coordinated systems.

---

## 🏛️ System Architecture
The system's intelligence is distributed among the participating nodes, which automatically form a computational chain and adapt to network changes.

### Key Components
- **Model Partitioning:**  
  The LLM is partitioned at the Transformer layer level. This simplifies the distribution logic, as each layer is a self-contained computational block.

- **Client Role:**  
  A client node initiates the inference task. It is responsible for the initial embedding layer and the final output layers. The client queries the DHT to find the head of the server chain and verify that a complete inference path is available.

- **Server Chain:**  
  Server nodes form a logical, sequential processing chain. Each server is aware of its successor, creating a clear data flow for the inference process.

- **Distributed Hash Table (DHT):**  
  Inspired by *Petals*, a DHT acts as the network's "bulletin board." Servers publish their ID, successor, the range of layers they host, and key performance metrics (memory, latency).

---

## ⚙️ Adaptive Layer Allocation Strategy
The framework employs a **three-phase, server-driven** strategy to manage the computational workload dynamically.

1. **Phase 1: Initial Chain Formation**  
   A new server discovers the "tail" of the chain via the DHT and attaches itself.  
   Servers greedily claim as many layers as their memory allows, announcing their assigned layers on the DHT.

2. **Phase 2: Adaptive Re-allocation**  
   After the first inference run, the network uses collected performance data (computational and communication delays) to re-balance the layer distribution.  
   Following the logic of **AR-MDI**, each server calculates its optimal share of the model's parameters based on its contribution to the network's total processing rate.

3. **Phase 3: Self-Healing & Opportunistic Takeover**  
   The network is designed to continuously improve.  
   - If a powerful new node joins, it can take over the role of a less performant node to increase overall throughput.  
   - If a node fails, the remaining servers coordinate through the DHT to redistribute the "orphaned" layers, ensuring the chain remains intact.

---

## 🛠️ Implementation Strategy
- **Language:** Python  
- **Core Library:** PyTorch  
- **P2P Networking:** hivemind  
- **Models:** Initial experiments will focus on manageable models like *Llama 3.1 8B*, using implementations from libraries such as `lit-gpt`.

---

## 📊 Evaluation
The framework's performance will be tested on a simulated heterogeneous environment using virtual machines or a collection of devices like Raspberry Pis and laptops.

**Metrics:**
- **Throughput:** Tokens per second.
- **Resource Utilization:** Distribution of memory and compute load across the network.

**Baselines for Comparison:**
1. A static, greedy partitioning of layers.
2. A naive allocation of an equal number of layers to each server.
3. Inference of a smaller LLM on a single device.
