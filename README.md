# Task1-Tagging_of_Thesis

#### Install Environment

```
$ python3 -m venv fund
$ . ./fund/bin/activate
$ pip -r requirements.txt
```


### Run Tensorboard Visualization
Run below command at the project root, then server will open at `http://localhost:6006`
```
$ tensorboard --logdir saved/log/
```

### Training

```
$ python train.py -c config.json -d 0
```

### Testing

```
$ python test.py -c config.json -d 0 --resume saved/models/ThesisTagging/1010_013058/model_best.pth
```