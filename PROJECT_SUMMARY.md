# Hybrid Python–Rust Perception Architecture Report Summary

## 1. Overview
This project, developed by **Team XLR8**, presents an integrated machine learning-based perception system for autonomous driving. It utilizes a **hybrid Python–Rust architecture** to combine the flexibility of Python's ML ecosystem (PyTorch/YOLO) with the efficiency, safety, and concurrency of Rust. The system is designed for a 1:10 scale autonomous vehicle platform.

## 2. Problem Statement
**The Conflict:** Autonomous driving requires real-time perception (low latency) using heavy Deep Learning models.
*   **Python** is excellent for developing ML models but struggles with concurrency (GIL) and latency when running multiple networked models.
*   **Rust** offers high performance and memory safety but lacks the rich ML tooling of Python.
*   **Goal:** Create a unified framework that allows researchers to develop in Python while offloading critical data flow and synchronization to Rust, enabling high-throughput real-time perception.

## 3. Technical Solution: Hybrid Architecture
The core innovation is a **Shared-Memory Infrastructure** managed by Rust that orchestrates data between Python processes.

### Key Components:
1.  **SharedMessage (Rust Layer):** A low-level shared memory layer that handles atomic synchronization and data exchange.
2.  **Partial-Blocking Producer–Consumer Model:**
    *   **Non-blocking:** Fastest, but risks desynchronization.
    *   **Full-blocking:** Safe but slow (wait for all).
    *   **Partial-blocking (Chosen):** The producer advances if *at least one* consumer has started. This allows fast modules (Lane Detection) to run at high frequency while slower modules (Sign Recognition) process frames asynchronously without continuously stalling the pipeline.
3.  **Process Separation:** Each perception task (Lanes, Objects, Signs) runs in its own process, interacting only via the Rust-backend shared memory.

## 4. Perception Stack & Machine Learning
The system employs multiple specialized models rather than a single monolithic one.

*   **Object Detection:** **YOLOv11s** (You Only Look Once).
    *   Specialized models for: **Pedestrians**, **Traffic Signs**, **Traffic Lights**.
*   **Lane Detection:** Custom **geometric algorithm** (Edge detection + Hough transforms). Preferred over ML for speed and reliability in this specific setup.
*   **Data Flow:** RGB frames are captured and written to shared memory. Consumers read the latest frame, process it, and write back predictions.

### Datasets & Training strategies:
1.  **Fine-tuning on BDD100K:**
    *   Used initially. Good for general urban scenes.
    *   **Failure:** Performed poorly on the 1:10 scale car due to "domain gap" (real city vs. toy city textures/lighting).
2.  **Custom Domain-Specific Dataset:**
    *   Recorded on the actual scale vehicle.
    *   Handcrafted and annotated.
    *   **Result:** Significantly higher reliability and drastically reduced false positives (e.g., stopping identifying chairs as traffic signs).

## 5. Performance & Results
*   **Frame Rate:** Achieved **35–90 FPS** on combined workloads (depending on hardware: Jetson AGX Orin / Desktop RTX 3090).
*   **Latency:** Reduced inter-process latency by up to **75%** compared to standard Python multiprocessing.
*   **Accuracy:** High detection reliability after custom dataset training.
    *   *Note:* BDD100K models had precision ~0.60 but poor real-world transfer.
    *   Custom models successfully differentiated traffic light states (Red/Green/Amber).

## 6. SWOT Analysis
*   **Strengths:** Hybrid performance, modularity (easy to swap models), real-time capability on embedded hardware.
*   **Weaknesses:** Custom dataset is small/narrow, manual annotation burden, lack of temporal consistency (tracking) in current version.
*   **Opportunities:** Adding semantic segmentation, using synthetic data (simulations), model compression.
*   **Threats:** Rapid AI evolution (VLMs replacing CNNs), OS-specific dependencies (Linux optimizations).

## 7. Conclusion
The project successfully demonstrates that a **hybrid architecture** is a viable path for bridging the gap between ML research prototype and real-time deployment. It proves that system-level optimization (Rust IPC) is just as critical as model accuracy for autonomous agents.
