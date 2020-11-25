# Pretraining XLNet 

This repo contains a working notebook for you to be able to pretrain XLNet LM from scratch with relative ease using an AI Platform Notebook instance. 

## Dataset
We will be using Dialpad Support Calls are from `talkiq-data.ai_research.dialpad_support_call_transcripts`

## Setup & Installation

#### Create AI Platform Notebook:

1. Follow the instructions from the AI Platform notebook guide [here](https://docs.google.com/document/d/1bP7MHUs9D8c3oBc9xEcljLtHJhzNm-TveX0dp6q0D_4/edit?usp=sharing) with a necessary step below
    * **\*\*IMPORTANT**** Make sure to create an instance with **CUDA 10.0** installed. This can be done by selecting the following option when creating an instance
        * ![cuda10_instance_example](img/env_img.png)
        
    * If not using AI Platform notebook, make sure your environment is setup with CUDA 10.0 and CuDNN 7.4 installed. 

2. Clone the repo:
    
    git clone https://github.com/matthiasdialpad/xlnet_pretrain.git

3. Next, install the necessary packages with:

    pip install -r requirements.txt
   
Now, you should be able to run `lm_pretrain.ipynb`

#### Overview to pretrain XLNet LM from scratch:
1. Decide on dataset you want to pretrain XLNet on 
2. Train SentencePiece tokenizer model on dataset (OR use an existing trained SentencePiece tokenizer model)
3. Process the dataset with the trained SentencePiece tokenizer using `xlnet/data_utils.py` 
4. We are now ready to pretrain an XLNet using `xlnet/train_gpu.py`

## XLNet Folder

This folder is cloned directly from XLNet author's GitHub

    git clone https://github.com/zihangdai/xlnet.git
    
As of this writing, I made minor modifications in `data_utils.py`, assigning `vocab_size=16000` instead of 32000 used by the authors.

#### CUDA
1. Make sure you have a GPU that is available by using `nvidia-smi` in terminal
2. Make sure tensorflow recognizes ur GPU as well. You can do this with:
```    
    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Cannot recognize GPU")
```        