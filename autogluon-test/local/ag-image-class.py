# image classification example
# reference
# https://autogluon.mxnet.io/tutorials/image_classification/beginner.html#

import autogluon
from autogluon import ImageClassification as task


train_data = task.Dataset(name='FashionMNIST')
test_data = task.Dataset(name='FashionMNIST', train=False)

# classification task
classifier = task.fit(train_data,
                      epochs=5,
                      ngpus_per_trial=1,
                      verbose=False)

# see the top-1 accuracy
print('Top-1 val acc: %.3f' % classifier.results['best_reward'])

# evaluate on test set
test_acc = classifier.evaluate(test_data)
print('Top-1 test acc: %.3f' % test_acc)
