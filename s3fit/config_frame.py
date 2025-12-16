# Copyright (C) 2025 Xiaoyang Chen - All Rights Reserved
# Licensed under the GNU GENERAL PUBLIC LICENSE Version 3
# Repository: https://github.com/xychcz/S3Fit
# Contact: s3fit@xychen.me

import numpy as np
np.set_printoptions(linewidth=10000)

class ConfigFrame(object):
    def __init__(self, config):
        self.config = config
        self.comp_c = [*config]
        self.num_comps = len(config)
        self.num_pars_c = [len(config[self.comp_c[i_comp]]['pars']) for i_comp in range(self.num_comps)]
        self.num_pars_c_max = max(self.num_pars_c)
        self.num_pars_c_tot = self.num_pars_c_max * self.num_comps # total number of pars of all comps (including placeholders)
        self.num_pars = self.num_pars_c_tot

        self.par_min_cp = np.full((self.num_comps, self.num_pars_c_max), -9999, dtype='float')
        self.par_max_cp = np.full((self.num_comps, self.num_pars_c_max), +9999, dtype='float')
        self.par_tie_cp = np.full((self.num_comps, self.num_pars_c_max), 'None', dtype='<U256')
        self.par_name_cp = np.array([['Empty_'+str(i_par) for i_par in range(self.num_pars_c_max)] for i_comp in range(self.num_comps)]).astype('<U256')
        self.par_index_cp = [] # index of each par in each comp
        self.info_c = [] 

        for i_comp in range(self.num_comps):
            par_index_p = {}
            for i_par in range(self.num_pars_c[i_comp]):
                if isinstance(config[self.comp_c[i_comp]]['pars'], list):
                    self.par_min_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][i_par][0]
                    self.par_max_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][i_par][1]
                    self.par_tie_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][i_par][2]
                elif isinstance(config[self.comp_c[i_comp]]['pars'], dict):
                    par_name = [*config[self.comp_c[i_comp]]['pars']][i_par]
                    par_index_p[par_name] = i_par
                    self.par_name_cp[i_comp,i_par] = par_name
                    if isinstance(config[self.comp_c[i_comp]]['pars'][par_name], list):
                        self.par_min_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][par_name][0]
                        self.par_max_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][par_name][1]
                        self.par_tie_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][par_name][2]
                    elif isinstance(config[self.comp_c[i_comp]]['pars'][par_name], dict):
                        self.par_min_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][par_name]['min']
                        self.par_max_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][par_name]['max']
                        self.par_tie_cp[i_comp,i_par] = config[self.comp_c[i_comp]]['pars'][par_name]['tie']
            self.par_index_cp.append(par_index_p)
                    
            self.info_c.append(config[self.comp_c[i_comp]]['info'])
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


