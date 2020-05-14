# reference
# https://autogluon.mxnet.io/tutorials/course/distributed.html?highlight=gpu

# 환경
# Cloud : GCP
# GPU : k80 x 1
# CPU : v4 RAM 20GB
# Jupyter notebook 에서 실행할 코드

import time
import numpy as np
import autogluon as ag

@ag.args(
     batch_size=64,
     lr=ag.Real(1e-4, 1e-1, log=True),
     momentum=0.9,
     wd=ag.Real(1e-4, 5e-4),
    )
def train_fn(args, reporter):
    print('task_id: {}, lr: {}'.format(args.task_id, args.lr))
    for e in range(10):
        top1_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        reporter(epoch=e, accuracy=top1_accuracy)
    # wait for 1 sec
    time.sleep(1.0)

# 나의 내부 ip 주소
extra_node_ips = ['172.31.3.95']

# Create a dist-learning scheduler. If no ip addresses are provided, the scheduler will only use local resources.
scheduler = ag.scheduler.FIFOScheduler(
    train_fn,
    resource={'num_cpus': 2, 'num_gpus': 1},
    dist_ip_addrs=extra_node_ips)

print(scheduler)
# FIFOScheduler(
# DistributedResourceManager{
# (Remote: Remote REMOTE_ID: 0,
#     <Remote: 'inproc://172.31.8.238/13943/1' processes=1 threads=8, memory=64.39 GB>, Resource: NodeResourceManager(8 CPUs, 0 GPUs))
# (Remote: Remote REMOTE_ID: 1,
#     <Remote: 'tcp://172.31.3.95:8702' processes=1 threads=8, memory=64.39 GB>, Resource: NodeResourceManager(8 CPUs, 0 GPUs))
# })

scheduler.run(num_trials=20)
scheduler.join_jobs()
# task_id: 1, lr: 0.0019243442240350372
# task_id: 2, lr: 0.012385569699754519
# task_id: 3, lr: 0.003945872233665647
# task_id: 4, lr: 0.01951486073903548
# [ worker 172.31.3.95 ] : task_id: 5, lr: 0.0006863718061933437
# [ worker 172.31.3.95 ] : task_id: 6, lr: 0.0016683650246923202
# [ worker 172.31.3.95 ] : task_id: 8, lr: 0.002783313777111095
# [ worker 172.31.3.95 ] : task_id: 7, lr: 0.0007292676946893176
# task_id: 9, lr: 0.08801928898220206
# task_id: 10, lr: 0.00026549633634006164
# task_id: 11, lr: 0.0009921995657417575
# task_id: 12, lr: 0.08505721989904058
# [ worker 172.31.3.95 ] : task_id: 13, lr: 0.04110913307416062
# [ worker 172.31.3.95 ] : task_id: 14, lr: 0.011746795144325337
# [ worker 172.31.3.95 ] : task_id: 15, lr: 0.007642844613083028
# [ worker 172.31.3.95 ] : task_id: 16, lr: 0.027900984694448027
# task_id: 17, lr: 0.018628729952415407
# task_id: 18, lr: 0.08050303425485368
# [ worker 172.31.3.95 ] : task_id: 19, lr: 0.0011754365928443049
# [ worker 172.31.3.95 ] : task_id: 20, lr: 0.008654237222679136

scheduler.get_training_curves()