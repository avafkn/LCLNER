# Label Consistency Loss for Multi-label Decoding in Name Entity Recognition

##1.Environments
-python (1.12.0)
-cuda(10.2)
##2.Dependencies
- numpy (1.21.4)
- torch (1.10.0)
- gensim (4.1.2)
- transformers (4.13.0)
- pandas (1.3.4)
- scikit-learn (1.0.1)
- prettytable (2.4.0)
##Dataset
CADEC
Conll 2003
## Training
#base
>> python main.py --config ./config/example.json
#base+LCL
>> python main_absloss.py --config ./config/example.json
#base+FLCL
>> python main_absloss_focalloss.py --config ./config/example.json



