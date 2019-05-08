# npi_bubblesort
Nerual program interpreters of bubblesort

### Authors(List by name)
* Bharat Matai

This project is based on these main resources: This project is establishing on Bucket Sort which will use bubble sort to sort the different buckets in and then collabrate all the buckets to form the results in the TestModel.py file the buckets are created and then sorted using the NPI MOdel using bubble sort, bubble sort code we try to establish from thr paper but are facing errors in giving accurate results and with Keras Implementation

* Roshni 
For multipication : Npi3 file is used which will establish the multipication between two numbers in the range of 1 to 1000 and trying to establish it. With the left pointer 

* Khyati
Nerual program interpreters of bubblesort


This project is based on these main resources:

Reference: https://github.com/mokemokechicken/keras_npi

pip install -r requirements.txt
```
tips: Keras use theano as backend. If you use tensorflow, it maybe many problems.

create training dataset
-----------------------
### create training dataset
```
sh src/run_create_bubblesort_data.sh
```

### create training dataset with showing steps on terminal
```
DEBUG=1 sh src/run_create_bubblesort_data.sh
```

training model
------------------
### Create New Model (-> remove old model if exists and then create new model)
```
NEW_MODEL=1 sh src/run_train_bubblesort_model.sh
```

### Training Existing Model (-> if a model exists, use the model)
```
sh src/run_train_bubblesort_model.sh
```

test model
----------
### check the model accuracy
```
sh src/run_test_bubblesort_model.sh
```
### check the model accuracy with showing steps on terminal
```
DEBUG=1 sh src/run_test_bubblesort_model.sh
```
