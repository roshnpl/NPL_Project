NPI Bucket sort using Bubble sort


Reference: https://github.com/mokemokechicken/keras_npi

requirement
-----------

* Python3

setup
-----

```
pip install -r requirements.txt
```
Keras use theano as backend. 


create training dataset
-----------------------
python create_training_data.py "train.pkl"


training model
------------------

python training_model.py "train.pkl" "bubble.model"


test model
----------

python test_model.py "bubble.model"
