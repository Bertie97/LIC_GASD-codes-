# Estimation of Joint Local Intensity Distribution for Multi-Modal Image Registration

This repository is the official implementation of the article ["*Estimation of Joint Local Intensity Distribution for Multi-Modal Image Registration*"](). 

## Requirements

To install requirements, please use `pip` to install: 

```sh
pip install -r requirements.txt
```

One can directly use the `brainweb` dataset provided in the repository, or fetch new ones on its [website](https://brainweb.bic.mni.mcgill.ca/brainweb/). 

For the `MS-CMR` dataset, one should apply the dataset on its [homepage](https://zmiclab.github.io/zxh/0/mscmrseg19/). 

## Contribution

The major implementations of LIC and GASD are provided in the file `color_transfer.py`. 

## Running

One can inspect the `*.log` files and use the `*.ckpt` files consisting of the state dictionaries for `Exp3` in Table 4, manuscript. 

To run the experiments, one should use the following command, 

```sh
cd Exp124\ LIC_GASD
python3 Exp1_LIC_GASD.py --exp <experiment name>
```

for experiments in Tables 1 and 2. 

Use commands, 

```sh
cd Exp124\ LIC_GASD
python3 Exp1_LIC_GASD_same_modality.py --exp <experiment name>
```

for the experiments in Table 3. 

The `experiment names` are listed in the dictionary `Experiments` in the Python files. 

One can use the following options to design the experiments on his own. 

```python
--n_dim			# the spatial dimension of the used dataset
--loss			# the loss function; selected from LIC, GASD_only, SSD, NVI, MI, NMI, NCC, or the local versions including NVI_local, MI_local, NCC_local, or the losses after image translation: GASD, CTr_LIC
--n_batch		# the batch size for paralleled processing
--n_iter		# the number of iterations (for the iterative method)
--sig_p			# the strength of the partial volume effect, $sigma_p$
--noise			# the strength of noise added on input images
--down_sample	# the downscaling factor
--exp			# experiment name
```

---

Use commands,

```sh
cd Exp3_LIC_deep
python3 Exp3_LIC_deep.py --exp <experiment name>
```

for the experiments in Table 4. 

One can design the experiments on his own by the following new options. 

```python
--n_epoch		# the number of training epochs
--n_epoch_save	# the period of model saving, in the unit of epoch numbers
--offset_scale	# the scaling of the predicted displacements
```

## Contributing

This code is under CC BY 4.0 licence. 
