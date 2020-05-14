# reference
# https://autogluon.mxnet.io/tutorials/course/distributed.html?highlight=gpu

# 환경
# Cloud : GCP
# GPU : k80 x 1
# CPU : v4 RAM 20GB
# Jupyter notebook 에서 실행할 코드
# working on "remote communication - connection refused"


import time
# import numpy as np
import autogluon as ag
from autogluon import ImageClassification as task


@ag.args(
     batch_size=64,
     lr=ag.Real(1e-4, 1e-1, log=True),
     momentum=0.9,
     wd=ag.Real(1e-4, 5e-4),
    )

def train_fn(args, reporter):
    print('task_id: {}, lr: {}'.format(args.task_id, args.lr))
    for e in range(10):

        # 여기에 학습 코드
        # Load data
        train_data = task.Dataset(name='FashionMNIST')
        test_data = task.Dataset(name='FashionMNIST', train=False)

        # classification task
        classifier = task.fit(train_data,
                              epochs=5,
                              ngpus_per_trial=1,
                              verbose=False)

        # see the top-1 accuracy
        print('Top-1 val acc: %.3f' % classifier.results['best_reward'])

        # top1_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
        top1_accuracy = classifier.results['best_reward']
        reporter(epoch=e, accuracy=top1_accuracy)

        # evaluate on test set
        test_acc = classifier.evaluate(test_data)
        print('Top-1 test acc: %.3f' % test_acc)

    # wait for 1 sec
    time.sleep(1.0)

# 나의 내부 ip 주소
extra_node_ips = ['xxx']

# Create a dist-learning scheduler. If no ip addresses are provided, the scheduler will only use local resources.
scheduler = ag.scheduler.FIFOScheduler(
    train_fn,
    resource={'num_cpus': 2, 'num_gpus': 0},
    dist_ip_addrs=extra_node_ips)

print(scheduler)

scheduler.run(num_trials=20)
scheduler.join_jobs()

scheduler.get_training_curves()
