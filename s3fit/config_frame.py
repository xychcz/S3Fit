# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np

class ConfigFrame(object):
    def __init__(self, config):
        self.config = config
        self.comp_c = [*config.keys()]
        self.num_comps = len([*config])
        self.num_pars_per_comp = len(config[[*config][0]]['pars'])
        self.num_pars = self.num_pars_per_comp * self.num_comps # total number of pars of all comps

        self.min_cp = np.zeros((self.num_comps, self.num_pars_per_comp), dtype='float')
        self.max_cp = np.zeros((self.num_comps, self.num_pars_per_comp), dtype='float')
        self.tie_cp = np.zeros((self.num_comps, self.num_pars_per_comp), dtype='<U64')
        self.info_c = [] # do not initialize string array for unknown string length, e.g., dtype='<U256'
        for i_comp in range(self.num_comps):
            for i_par in range(self.num_pars_per_comp):
                self.min_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][0]
                self.max_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][1]
                self.tie_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][2]
            self.info_c.append(config[[*config][i_comp]]['info'])
            self.info_c[i_comp]['comp_name'] = [*config.keys()][i_comp]
        # self.info_c = np.array(self.info_c)

    def flat_to_arr(self, vals_flat):
        return vals_flat.reshape(self.num_comps, int(len(vals_flat)/self.num_comps))
