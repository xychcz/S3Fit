# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Contact: xiaoyang.chen.cz@gmail.com

import numpy as np

class ConfigFrame(object):
    def __init__(self, config):
        self.config = config
        self.comp_c = [*config.keys()]
        self.num_comps = len([*config])
        self.num_pars = len(config[[*config][0]]['pars'])
        self.num_compxpars = self.num_comps * self.num_pars # used for flattened input parameters of fitting
        self.min_cp = np.zeros((self.num_comps, self.num_pars), dtype='float')
        self.max_cp = np.zeros((self.num_comps, self.num_pars), dtype='float')
        self.tie_cp = np.zeros((self.num_comps, self.num_pars), dtype='<U64')
        self.info_c = [] # do not initialize string array for unknown string length, e.g., dtype='<U256'
        for i_comp in range(self.num_comps):
            for i_par in range(self.num_pars):
                self.min_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][0]
                self.max_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][1]
                self.tie_cp[i_comp,i_par] = config[[*config][i_comp]]['pars'][i_par][2]
            self.info_c.append(config[[*config][i_comp]]['info'])
            self.info_c[i_comp]['comp_name'] = [*config.keys()][i_comp]
        # self.info_c = np.array(self.info_c)

    # def flat_to_arr(self, pars_flat):
    #     return pars_flat.reshape(self.num_comps, self.num_pars)
    def flat_to_arr(self, vals_flat):
        return vals_flat.reshape(self.num_comps, int(len(vals_flat)/self.num_comps))
