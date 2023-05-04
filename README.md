# Label Consistency Loss for Multi-label Decoding in Name Entity Recognition

##1.Environments
```
-python (1.12.0)
-cuda(10.2)
```


##2.Dependencies
```
same as W2NER,maybe in different version

```
##Dataset

```
CADEC
Conll 2003
```
## Training
```
base
python main.py --config ./config/example.json

base+LCL
python main_absloss.py --config ./config/example.json

base+FLCL
python main_absloss_focalloss.py --config ./config/example.json
```



