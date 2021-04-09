wget https://apt.repos.intel.com/openvino/2021/GPG-PUB-KEY-INTEL-OPENVINO-2021
sudo apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2021
echo "deb https://apt.repos.intel.com/openvino/2021 all main" | sudo tee /etc/apt/sources.list.d/intel-openvino-2021.list
sudo apt update -y
sudo apt install intel-openvino-runtime-ubuntu18-2021.3.394 -y
git clone https://github.com/openvinotoolkit/open_model_zoo.git
python open_model_zoo/tools/model_downloader/downloader.py --name bert-large-uncased-whole-word-masking-squad-0001
