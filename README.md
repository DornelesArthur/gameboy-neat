🇺🇸 English | 🇧🇷 [Português](README.pt-br.md)

# GameBoy NEAT Cluster Analysis

## 🧠 About the Project
This project implements a distributed cluster system to train neural networks using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. The simulation environment is the *Super Mario Land* Game Boy game, running on the open-source **PyBoy** emulator. 

Originally validated on an AWS EC2 cloud environment, the main goal of this architecture is to perform scalability and performance tests by parallelizing the neural network training across multiple nodes.

## ⚙️ System Architecture
The cluster uses a Manager-Worker architecture, communicating asynchronously via TCP/IP sockets:
* **Manager Node:** Orchestrates the NEAT algorithm, controlling populations, reproduction, and mutations. It serializes genomes into bytes and distributes them to the trainers via separate threads.
* **Trainer Nodes:** Parallel computing nodes dedicated exclusively to running the PyBoy emulator. They receive the genome, instantiate the neural network, convert game screen pixels into Numpy arrays for network inputs, and return the fitness score.

## 🚀 Local Execution (Docker)

### Prerequisites
Before building the images, ensure the original game ROM is placed in the repository's root directory with the exact name expected by the `Dockerfile`:
* `SuperMarioLand.gb`

### Running the Cluster
Create a local bridge network to allow TCP/IP communication between containers:
```bash
docker network create test-bridge

```

**1. Start the Manager**
Build and run the Manager container, fixing the IP so trainers can connect to the established socket:

```bash
DOCKER_BUILDKIT=0 docker build -t managerdocker --network=test-bridge -p 37259:37259 -f manager.dockerfile .
docker run -t managerdocker -p 37259:37259 --name test -rm --ip 172.17.0.2

```

**2. Start the Trainer(s)**
In a separate terminal session in your local Fedora environment, build and connect the Trainer node(s) to the same network:

```bash
DOCKER_BUILDKIT=0 docker build --network=test-bridge -t trainerdocker -f trainer.dockerfile .
docker run -t trainerdocker -p 37259:37259 --name testtrainer -rm

```

## 📊 Performance Results

The architecture was tested to evaluate performance bottlenecks and optimization strategies:

* **Parallelization:** Using a 5-trainer cluster setup, the average training time per generation was reduced by a factor of ~4.63x. This demonstrates the system scales efficiently with the addition of new computational nodes.


* **Dimensionality Reduction:** Scaling down the pixel input resolution significantly decreased the neural network processing time. Training with the standard 160x144 resolution (23040 inputs) took an average of ~191 seconds, while a reduced 54x48 resolution (2592 inputs) took only ~24 seconds (7.8x faster).
