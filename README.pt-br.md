🇺🇸 [English](README.md) | 🇧🇷 Português

# Análise de Performance NEAT em Cluster Distribuído

## 🧠 Sobre o Projeto
Este projeto implementa um sistema distribuído (cluster) projetado para treinar redes neurais utilizando o algoritmo genético **NEAT (NeuroEvolution of Augmenting Topologies)**. O ambiente de simulação escolhido foi o jogo *Super Mario Land* de Game Boy, executado através do emulador open-source **PyBoy**.

Validado originalmente em um ambiente de nuvem (AWS EC2), o objetivo principal desta arquitetura é viabilizar testes de performance e análise de escalabilidade paralelizando o treinamento de redes neurais através de múltiplos nós.

## ⚙️ Arquitetura do Sistema
O cluster utiliza uma arquitetura baseada no modelo Gerente-Trabalhador (*Manager-Worker*), comunicando-se via sockets TCP/IP:
* **Nó Gerente (Manager):** Responsável por orquestrar o algoritmo NEAT, gerenciando as populações, ciclos de reprodução e mutação. Ele serializa os genomas em bytes e os distribui para os nós de treinamento de forma assíncrona usando threads.
* **Nós Treinadores (Trainers):** Nós de computação paralela dedicados exclusivamente à execução do emulador. Eles recebem o genoma, instanciam a rede neural, convertem os pixels da tela do jogo em matrizes Numpy para as entradas da rede e retornam o *fitness score* (pontuação de aptidão) ao Gerente.

## 🚀 Execução Local com Docker

### Pré-requisitos
Antes de construir as imagens, certifique-se de que a ROM original do jogo esteja na raiz do repositório com o nome exato esperado pelo `Dockerfile`:
* `SuperMarioLand.gb`

### Subindo o Cluster
Crie a rede bridge local para permitir a comunicação TCP/IP entre os contêineres:
```bash
docker network create teste-bridge

```

**1. Iniciar o Gerente (Manager)**
Construa e rode o contêiner do Gerente fixando o IP, para que os treinadores consigam encontrá-lo e estabelecer os sockets de conexão:

```bash
DOCKER_BUILDKIT=0 docker build -t managerdocker --network=teste-bridge -p 37259:37259 -f manager.dockerfile .
docker run -t managerdocker -p 37259:37259 --name teste -rm --ip 172.17.0.2

```

**2. Iniciar o(s) Treinador(es)**
Em um terminal separado no seu ambiente Fedora local, construa e conecte os nós Treinadores à mesma rede do Gerente:

```bash
DOCKER_BUILDKIT=0 docker build --network=teste-bridge -t trainerdocker -f trainer.dockerfile .
docker run -t trainerdocker -p 37259:37259 --name testetrainer -rm

```

## 📊 Resultados de Performance

A arquitetura foi submetida a testes rigorosos para avaliar gargalos e estratégias de otimização:

* **Paralelização:** Utilizando um setup com 5 treinadores, o tempo médio de treinamento por geração foi reduzido em um fator de ~4.63x. Isso comprova que o sistema escala de forma eficiente com a adição de novos nós.


* **Redução de Dimensionalidade:** A diminuição da resolução dos pixels de entrada reduziu drasticamente o tempo de processamento das redes neurais. O treinamento com a resolução padrão de 160x144 (23040 entradas) levou em média ~191 segundos, enquanto uma resolução reduzida de 54x48 (2592 entradas) levou apenas ~24 segundos (7,8x mais rápido).
