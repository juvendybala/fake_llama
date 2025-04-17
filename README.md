# fake_llama
## Introduction
这个仓库主要记录了南京大学计算机学院本科生课程智能应用开发的课程项目，该课程的授课老师是[徐经纬](https://ics.nju.edu.cn/people/jingweixu/)教授。这个项目实现了一个在结构和功能上基本与`llama 3`等价的大语言模型，涉及到较多大语言模型相关的论文和技术，也涉及到较多大语言模型的具体实现细节，这个仓库希望能够给有志于从事大语言模型的同学一个参考和启发。该项目对应的课程为[智能应用开发](https://njudeepengine.github.io/llm-course-lecture)，对应的作业地址为[open-llm-assignments](https://github.com/NJUDeepEngine/open-llm-assignments)，你也可以将这个仓库看做一个课程作业的参考。如果你对于这门课程感兴趣，可以关注教师的b站账号，UID为`390606417`，2024年秋季的线上课程录像可以在[这里](https://space.bilibili.com/390606417/channel/collectiondetail?sid=3771310)获得。

## Environment
* 你需要提前安装好 python 3.10+的环境
* 在项目文件夹下运行命令`pip install -r requirements.txt`，即可以获得一个完整的项目环境
* 课程会提供已经配置好的`docker`环境，可以避免可能的冲突以及更轻松的进行开发和`debug`，但需要选上这门课或者联系老师或助教才可以获得

## Tasks
整个项目需要完成的内容都包括在`./tasks/*.md`中，其中有对于每个具体实现任务的阐释，后续对于项目文件和代码的介绍中也会再次提到这些任务。

## Tests
* 对于每个任务，都提供了一些最基本的测试单元，存放在`./tests`文件夹下。对于`test_toy_task[0-4].py`，可运行命令`pytest test_toy_task[0-4].py`测试实现的正确性；对于`test_toy_task5-[1-3].py`，可运行命令`python test_toy_task5-[1-3].py`测试实现的合理性
* 注意：对于前四个task的测试是使用了硬编码的策略，即输入的情况已经被定死，输出也已经硬编码与自己的实现进行比较，通过这个测试并不意味着自己的实现是正确的，如果想要更加全面的测试自己的实现，需要得到助教提供的`docker`镜像和作业[地址](https://github.com/NJUDeepEngine/open-llm-assignments)中的`test_with_ref.py`文件进行更加全面和灵活的测试
* 注意：最后一个task的测试是和`huggingface`上标准的`llama`实现进行对比，只关注合理性即可，因为很多具体的实现细节与`llama 3`并不相同，同样的，如果想更进一步的测试，需要需要得到助教提供的`docker`镜像，可以在`test_toy_task5-[1-3].py`和助教实现的黄金版本进行对比
* 注意：如果想对task5进行测试，需要在`./model/llama_3.2_***`文件夹下放置从`huggingface`上下载的模型权重和配置文件，还需要在`./data/chat`和`./data/qa`文件夹下放置数据集配置文件。**PS：目前项目代码只支持`llama_3.2_1b_instruct`模型在task5的运行，其余模型可能存在一些导入问题**
* 注意：对于`test_toy_task5-3.py`后半段`lora`训练的代码还存在一些问题，目前并没有公开到仓库，如果有同学有兴趣可以补全完成

## Project descrpition
这一部分主要介绍了该课程项目中我自己实现的部分，主要是简略介绍实现的模块所对应的功能和论文.相比于`./tasks`文件夹中详细介绍了实现的细节，这一部分能让你更加直观的了解到这个项目做了些什么，可能会引起你的兴趣

### `./data`
该文件夹下存储了两种不同的数据集，分别为`chat`数据集和`qa`数据集，他们会在`task5-3`中涉及

### `./model`
该文件夹下存储了不同的模型对应的参数和模型配置，可从`huggingface`中下载，他们主要在`task5-1`中涉及

 ### `./src/inference/agent.py`
 在这个文件中实现了一个`InferenceAgent`类，他可以接受一个大语言模型和相关的`prompt`以及`query`进行推理过程，可通过控制`InferenceConfig` 采用不同的推理策略进行推理，具体的实现细节可见[这里](./tasks/task5-2.md)

 ### `./src/training/lora.py`
 在这个文件中实现了一个简单的`LoraTrainer`类，可针对一个大语言模型进行`lora`训练，具体的实现细节可见[这里](./tasks/task5-3.md)

 ### `./src/modeling/datasets/`
 在该文件夹下的`chat.py`和`qa.py`中分别实现了两个类可以构建对应的数据集用于后续的`SFT`训练，具体的实现细节可见[这里](./tasks/task5-3.md)

 ### `./src/modeling/models/llama.py`
 在这个文件中实现了一个最基本的`llama`模型，他支持模型参数的导入和模型配置的导入，可以在训练和推理两个不同任务情况下进行前向传播，进行正确的`kv_cache`管理，涉及到的论文有[Llama3 Paper](https://arxiv.org/pdf/2407.21783)，具体的实现细节可见[这里](./tasks/task5-1.md)

### `./src/modeling/norm.py`
在这个文件中实现了`RMSNorm`的变种`GroupRMSNorm`类，可对输入进行分组`normalization`，涉及到的论文有[RMSNorm Paper](https://arxiv.org/abs/1910.07467)，具体的实现细节可见[这里](./tasks/task1-1.md)

### `./src/modeling/vocab_emb.py`
在这个文件中实现了一个`ParallelVocabEmbedding`类，它模拟了分布式情况下的`vocab_embedding`操作，主要功能是将离散的`token`转化为连续的`embedding`，涉及到的资料有[Megatron Vovab Parallel Embedding Module](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L156)，具体的实现细节可见[这里](./tasks/task1-2.md)

### `./src/modeling/pos_emb.py` and `./src/functional.py`
这两个文件共同实现了一个大语言模型中的重要模块：位置编码模块，当前普遍使用的位置编码类型为[RoPE](https://arxiv.org/abs/2104.09864)，即旋转位置编码，其功能是为序列的`embedding`加上位置信息。该项目实现了`Rope`的一个变种，[NTK-aware RoPE](https://arxiv.org/abs/2309.00071)，用于解决训练环境下和推理环境下文本长度不一致的问题，相关论文有[Alibi](https://arxiv.org/abs/2108.12409)和[PI](https://arxiv.org/abs/2306.15595)，具体的实现细节可见[这里](./tasks/task1-3.md)

### `./src/modeling/mlp.py`
在这个文件中实现了大语言模型中的`mlp`层，其主要功能是通过非线性变换提取高阶特征，生成更有利于下一层处理的特征表示。文件中主要实现了两个类，分别为`DenseMLPWithLoRA`和`SparseMLPWithLoRA`，前者实现了一个`mlp`模块，其中保存`lora`参数用于后续的训练，`lora`有关的论文见[这里]((https://arxiv.org/abs/2106.09685))，同时该模块支持多种非线性激活函数，相关论文有[GLU Paper](https://arxiv.org/abs/1612.08083)和[GLU Variants Paper](https://arxiv.org/abs/2002.05202)，具体的实现细节见[这里](./tasks/task2-1.md)；后者在前者的基础上实现了一个`MOE`风格的稀疏`MLP`层，同时它也模拟了分布式情况下`mlp`的操作，相关论文有[MoE Paper](https://arxiv.org/abs/1701.06538)和[Mixtral Paper](https://arxiv.org/abs/2401.04088)，该文件具体的实现细节见[这里](./tasks/task2-2.md)

### `./src/modeling/attention.py`
在这个文件中实现了大语言模型的`attention`层，有关`attention`层的结构和功能可见论文[Transformer paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)。这个文件实现了两个类，分别为`OfflineSlidingWindowAttn`和`OnlineSlidingWindowAttn`，前者实现了一个经典的`attention layer`，支持不同的`QKV pack format`，`QKV Layout`和`Window`模式，同时支持不同的多头注意力机制，包括`MHA`,`MQA`,`GQA`，相关论文有[Google MQA paper](https://arxiv.org/pdf/1911.02150)和[Google GQA paper](https://arxiv.org/pdf/2305.13245)，具体实现细节见[这里](./tasks/task3-1.md)；后者实现了一个支持`online softmax`的`attention layer`，`online softmax`是一个在`attention layer`层面非常重要的优化，可以减少`IO`操作和提供分布式的扩展，详情见[Online Softmax Paper](https://arxiv.org/pdf/2112.05682)，它也为`flash attention`提供了分块计算的基础，相关论文见[Flash Attention 2 Paper](https://arxiv.org/pdf/2307.08691.pdf)，该文件具体实现细节见[这里](./tasks/task3-2.md)

### `./src/modeling/transformer.py`
在这个文件中实现了大语言模型的`kv cache`，`transformer layer`和`transformer block`，第一个模块在大语言模型的推理中起到了很重要的作用，它可以在`decoding`阶段加速推理的速度，是一种以空间换时间的策略，具体实现细节见[这里](./tasks/task4-1.md)；第二个模块和第三个模块组成最终的大语言模型，是对上述实现的模块的拼接，具体实现细节见[这里](./tasks/task4-2.md)和[这里](./tasks/task4-3.md)

### `./src/modeling/prompt.py`
这个文件中实现了大语言模型中推理过程中的`prompt`类，它可用于大语言模型推理前给出对于模型生成的指示，具体实现细节见[这里](./tasks/task5-2.md)

## Learnings
* 如果你是一个从未接触过深度学习领域的初学者，这个项目可能可以让你对于`pytorch`的基本语法有一个了解，你可以学习到许多`pytorch`的语法糖以及如何对张量进行操作，进而提升代码能力
* 如果你是一个有志于从事大语言模型研究的研究者，这个项目能让你对于大语言模型有一个比较基本的认知，能够知道一个`llama`模型的每个部件都是在做些什么，以及一些具体的代码实现细节，能够让你知道大语言模型可能的科研方向和相关工作，是一个不错的初上手项目
* 如果你是一个从事于ai领域其他方向的研究者，这个项目实现的内容可能和你的研究方向并不直接相关，你可能往往将它当做一个黑盒来使用，但了解其内部的构造和运行过程可以让你更好地使用它