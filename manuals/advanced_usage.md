# Advanced usage

## Support new band filters

The transmission curve supported by S<sup>3</sup>Fit needs to have 
two columns, wavelengths (in angstrom) and transmission values.
Save the curve with a filename of `Bandname.dat` and put to the directory set in `phot_trans_dir`, 
and then the new band can be used in S<sup>3</sup>Fit. 

## Modify extinction laws

The default extinction law of S<sup>3</sup>Fit is [Calzetti00](http://www.bo.astro.it/~micol/Hyperz/old_public_v1/hyperz_manual1/node10.html).
If you would like to use another extinction law, please navigate to the `Extinction Functions` section of the S<sup>3</sup>Fit, 
define the the new extinction function that output $A_\lambda/A_V$, 
and remember to specify the new extinction law as the default one by modifying `ExtLaw = ExtLaw_NEW`. 

## Support new emission lines

Please navigate to the `set_linelist()` function in the `ELineModels` class in `s3fit.py` to add new emission lines.
The rest wavelengths and names are stored in the lists `line_rest_n` and `line_name_n`
(note that the rest wavelength is given in vacuum).
If lineA is tied to lineB with a fixed flux ratio, 
set `linked_to_n` of lineA to `line_rest_n` of lineB, and `linked_ratio_n` of lineA to the flux ratio of lineA/lineB. 
The follow coding block exhibits the example with [OIII] doublets, 
where [OIII]a is tied to [OIII]b with a flux ratio of 0.335
(please read the <ins>**Emission lines**</ins> section in [basic usage](manuals/basic_usage.md) for calculation of the flux ratio).  
```python
self.line_rest_n.append(5008.240); self.linked_to_n.append(-1)      ; self.linked_ratio_n.append(-1)     ; self.line_name_n.append('[OIII]b')
self.line_rest_n.append(4960.295); self.linked_to_n.append(5008.240); self.linked_ratio_n.append(0.335)  ; self.line_name_n.append('[OIII]a')
```

## Support new models

#### Create ModelFrame
```python
init
reduce if template norm
return to obs
```

#### Create configuration dictionary
```python

```

#### Add to FitFrame
load models
```python

```

#### Output related results
print best x coeff
plot best model
print function 

