# deepMNIST-distributed-example

<b>Objective</b>

There are different ways to distribute training, i.e. across machines and GPUs (as mentioned [here](http://stackoverflow.com/questions/37732196/tensorflow-difference-between-multi-gpus-and-distributed-tensorflow/37733117#37733117)). In this experiment we will perform replicated training across machines.

For the experiment we will build a simple architecture with 'workers' on two-three machines which will perform the actual training, and one 'parameter-server' (ps) which will synchronize parameters obtained during training. For more info see: 'How to Train from Scratch in a Distributed Setting' in [inception: How to Train from Scratch in a Distributed Setting](https://github.com/tensorflow/models/tree/master/inception).

<b>Observations</b>

* Setting up parameter server (with gpu enabled) and worker server on the same machine reduced computation. Hence mask the cuda device when running ps.
* Setting up parameter server and worker server on the same machine reduced communication overhead

<b>Hypothesis</b>
* More number of machines will compute faster, if task given is compute intensive

<b>Experiment</b>

<b> For training using stand alone deepMNIST code just run as: </b>

    python trainer_deepMnist_tensorboard_example.py

The model will get saved in folder `1` inside `summaries` folder. This can be used by tensorboard for visualizing the cross entropy loss and and test accuracy. For more info: [link](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html)

    #tensorboard --logdir=path/to/log-directory
    tensorboard --logdir=summaries/1

Then simply view in your browser at `localhost:6006`

<b> For training using distributed deepMNIST code follow these steps: </b>

Say you have two machines for distributing your training. At machine 1 it will run as 'ps with task id 0' by specifying the 'job_name' as 'ps' and 'task_index' as '0' using the following command:

    CUDA_VISIBLE_DEVICES='' python distributed_trainer_deepMnist_v2.py --ps_hosts=machine1:2223 --worker_hosts=machine1:2222,machine2:2222 --folder_no=2 --job_name=ps --weights_name=30Aug_1kepcohs_2machines --epochs=1000 --task_index=0

 If ps and worker are working on the same gpu it can lead to cuda out of memory error, hence while running ps the CUDA device should be masked using CUDA_VISIBLE_DEVICES=''

On machine 1 start the 'worker with task id 0' by specifying the 'job_name' as 'worker' and 'task_index' as '0' using the following command:


    python distributed_trainer_deepMnist_v2.py --ps_hosts=machine1:2223 --worker_hosts=machine1:2222,machine2:2222 --folder_no=2 --job_name=worker --weights_name=30Aug_1kepcohs_2machines --epochs=1000 --task_index=0

On machine 2 start the 'worker with task id 1' by specifying the 'job_name' as 'worker' and 'task_index' as '1' using the following command:


    python distributed_trainer_deepMnist_v2.py --ps_hosts=machine1:2223 --worker_hosts=machine1:2222,machine2:2222 --folder_no=2 --job_name=worker --weights_name=30Aug_1kepcohs_2machines --epochs=1000 --task_index=1

The same way you can specify more workers in the arguments and use more machines for distrbution. As soon as the parameter server starts on machine 1, worker 0 which is the 'chief' starts with the training and supervises it for the rest of the workers.

Here are some links which might be useful for better understanding:

* [stackoverflow: distributed-tensorflow](http://stackoverflow.com/questions/39027488/distributed-tensorflow-replicated-training-example-grpc-tensorflow-server-no)
* [Announcing TensorFlow 0.8 – now with distributed computing support!](https://research.googleblog.com/2016/04/announcing-tensorflow-08-now-with.html)
* [Leo K. Tam Blog](http://leotam.github.io/general/2016/03/13/DistributedTF.html)
* [Distributed TensorFlow: Scaling Google’s Deep Learning Library on Spark](https://arimo.com/machine-learning/deep-learning/2016/arimo-distributed-tensorflow-on-spark/)
* [Distributed TensorFlow Internal Architecture Summary](http://bettercstomorrow.com/2016/07/14/distributed-tensorflow-internal-architecture-summary/)
* [Imanol Schlag (github)](http://ischlag.github.io/2016/06/12/async-distributed-tensorflow/)
* [tensorflow git issue 4033](https://github.com/tensorflow/tensorflow/issues/4033#issuecomment-243163770)
* [Distributed tensorflow parameter server and workers](http://stackoverflow.com/questions/38185702/distributed-tensorflow-parameter-server-and-workers)


<b>Results</b>

Single machine computation was found to be faster at first, when each epoch was training on a batch of just 50 images. One reason was that, the communication overhead between machines increased time for computation of each epoch. Second reason was that, the task wasn't compute intensive, hence one machine with gpu ran faster computations. For the same reason another experiment was performed where, number of images taken in each epoch was increased (to 1000 images) and performance was evaluated again. In this second experiment the distributed tensorflow implementation (multiple machine with single GPUs) could compute the epochs faster than a single machine with a GPU. Though an implementation on one machine with multiple GPUs would be much faster as there will be no network latency.

<p align="center">
  <img src="https://github.com/abhijayghildyal/deepMNIST-distributed-example/blob/master/1000imagesPerIteration.png" align="middle" height="280" width="600" margin="0 auto" />
</p>


<p align="center">
  <img src="https://github.com/abhijayghildyal/deepMNIST-distributed-example/blob/master/benchmarking.png" align="middle" height="300" width="700" margin="0 auto" />
</p>
