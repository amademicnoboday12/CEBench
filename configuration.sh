pip3 install langchain==0.1.14
pip3 install langchain_community==0.0.31
pip3 install --upgrade qdrant-client
pip3 install langchainhub
pip3 install loguru
pip3 install -qU langchain-openai
pip3 install nvidia-ml-py3
pip3 install pandas
pip3 install paretoset

#########ollama
docker pull ollama/ollama
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

#########qdrant
docker pull qdrant/qdrant

mkdir -p knowledge_base llms results queries templates vec_db


# Run the containers before evaluation
docker run --rm -d --gpus=all -v $(pwd)/llms:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
docker run --rm -d -p 6333:6333 -v $(pwd)/vec_db:/qdrant/storage --name qdrant qdrant/qdrant
