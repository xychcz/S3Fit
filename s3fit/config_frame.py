# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import numpy as np
np.set_printoptions(linewidth=10000)

class ConfigFrame(object):
    def __init__(self, config):
        self.config = config
        self.comp_c = [*config.keys()]
        self.num_comps = len([*config])
        self.num_pars_per_comp = max([len(config[[*config][i_comp]]['pars']) for i_comp in range(self.num_comps)])
        self.num_pars = self.num_pars_per_comp * self.num_comps # total number of pars of all comps

        self.min_cp = np.full((self.num_comps, self.num_pars_per_comp), -9999, dtype='float')
        self.max_cp = np.full((self.num_comps, self.num_pars_per_comp), -9999, dtype='float')
        self.tie_cp = np.full((self.num_comps, self.num_pars_per_comp), 'fix', dtype='<U64')
        self.info_c = [] # do not initialize string array for unknown string length, e.g., dtype='<U256'
        for i_comp in range(self.num_comps):
            for i_par in range(len(config[[*config][i_comp]]['pars'])):
                self.min_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][0]
                self.max_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][1]
                self.tie_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][2]

            self.info_c.append(config[[*config][i_comp]]['info'])
            # group used model elements in a list
            for item in ['mod_used', 'line_used']:
                if np.isin(item, [*self.info_c[i_comp]]): 
                    if isinstance(self.info_c[i_comp][item], list): 
                        self.info_c[i_comp][item] = np.array(self.info_c[i_comp][item])
                    else:
                        self.info_c[i_comp][item] = np.array([self.info_c[i_comp][item]])

            # rename sign for absorption/emission
            if np.isin('sign', [*self.info_c[i_comp]]):
                if np.isin(self.info_c[i_comp]['sign'], ['absorption', 'negative', '-']):
                    self.info_c[i_comp]['sign'] = 'absorption'
                if np.isin(self.info_c[i_comp]['sign'], ['emission', 'positive', '+']):
                    self.info_c[i_comp]['sign'] = 'emission'
            else:
                self.info_c[i_comp]['sign'] = 'emission' # default

            self.info_c[i_comp]['comp_name'] = [*config.keys()][i_comp]
        # self.info_c = np.array(self.info_c)

    def flat_to_arr(self, vals_flat):
        return vals_flat.reshape(self.num_comps, int(len(vals_flat)/self.num_comps))