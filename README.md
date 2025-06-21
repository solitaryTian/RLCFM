强化学习PCM代码。sh文件是运行指令，主文件是train_pcm_base_model_sdxl_adv_RL.py，跟强化学习最相关的内容在此文件的class Q_learning中。

1-基础环境准备：

conda create -n RLPCM python=3.10 

conda activate RLPCM

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 

2-手动安装核心库diffusers

git clone https://github.com/huggingface/diffusers

cd diffusers

pip install -e .


3-启动训练代码

bash train_pcm_base_model_sdxl_RL_dmd.sh

如果想运行此代码，还需要准备数据集
