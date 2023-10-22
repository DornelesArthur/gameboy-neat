# gameboy-neat

## Manager:
DOCKER_BUILDKIT=0 docker build -t managerdocker --network=teste-bridge -p 37259:37259 -f manager.dockerfile .

docker run -t managerdocker -p 37259:37259 --name teste -rm --ip 172.17.0.2

## Trainer:
DOCKER_BUILDKIT=0 docker build --network=teste-bridge -t trainerdocker -f trainer.dockerfile .

docker run -t trainerdocker -p 37259:37259 --name testetrainer -rm

