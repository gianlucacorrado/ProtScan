# ProtScan
Protein-RNA target site modelling and prediction.

Installation
------------
First install EDeN 1.0 (for EDeN installation, please refer to [EDeN repository](https://github.com/fabriziocosta/EDeN/releases/tag/v1.0)) and the other dependencies:
```
joblib (tested on v. 0.9.4)
numpy (tested on v. 1.11.2)
scipy (tested on v. 0.18.1)
```

Then, clone the ProtScan repository
```
git clone https://github.com/gianlucacorrado/ProtScan.git
```
Finally, exectute the setup script
```
cd ProtScan
python setup.py install
```

Documentation
=============
ProtScan allows 5 subcommands:
```
optimize            Optimize model hyperparameters (and fit --fit-opt).
fit                 Fit a model.
predict             Predict binding affinities using a model.
crosspredict        2-fold cross validated predictions.
extractpeaks        Extract peaks from a predicted profile.
```

Optimize
--------
The optimize subcommand allows to optimize the model hyperparameters, and, if wanted, to also fit a model with the optimized
hyperparameters.

The optimization subcommand requires to specify: the fasta file with the RNA sequences, the BED file with the protein binding sites, the name of the model where to save the optimal hyperparameters.
```
protscan optimize -f <fasta_file> -b <bed_file> -m <model_name>
```

For the full argumets list, refer to the help of the optimize subcommand:
```
protscan optimize -h
```

Fit
---
The fit subcommand allows to fit a model. It can be done using the default hyperparameters, or using the ones obtained with the optimization step.

The fit subcommand requires to specify: the fasta file with the RNA sequences, the BED file with the protein binding sites, the name of the model where to save the model or the name of a model with with the hyperparameters optimized by the optimize subcommand.
```
protscan fit -f <fasta_file> -b <bed_file> -m <model_name>
```

For the full argumets list, refer to the help of the fit subcommand:
```
protscan fit -h
```

Predict
-------
The predict subcommand allows to use a fitted model to predict the interaction affinity on a set of RNAs.

The predict subcommand requires to specify: the fasta file with the RNA sequences, the name fitted the model to use for the prediction, the output file where to save the predicted profiles (cPickle):
```
protscan predict -f <fasta_file> -m <model_name> -o <out_pkl_file>
```

For the full argumets list, refer to the help of the predict subcommand:
```
protscan predict -h
```

Crosspredict
------------
The crosspredict subcommand allows predfict the interacton affinity for a set of RNAs using a 2-fold cross-validation procedure.

The crosspredict subcommand requires to specify: the fasta file with the RNA sequences, the BED file with the protein binding sites, the output file where to save the predicted profiles (cPickle), and optionally the name of the model with the optimized hyperparameters:
```
protscan crosspredict -f <fasta_file> -b <bed_file> -o <out_pkl_file> (-m <model_name>)
```

For the full argumets list, refer to the help of the crosspredict subcommand:
```
protscan crosspredict -h
```

Extractpeaks
------------
The extractpeaks subcommand allows select areas of the RNA showing statistically significant enrichment is the affinity with the protein.

The extract subcommand requires to specify: the file containing the predicted profiles (cPickle), the BED file with the protein binding sites, the output file where to save the predicted profiles (BED):
```
protscan crosspredict -p <profiles> -b <bed_file> -o <out_bed_file> 
```

For the full argumets list, refer to the help of the crosspredict subcommand:
```
protscan extractpeaks -h
```
