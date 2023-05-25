import numpy as np
import pandas as pd

import looti.tools as too

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle



class objdict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)



class DataHandle:
    """
    Class handling the extraction of data from pandas databases 
    and the separation of data into training, validation and test sets.

    Args:
       extmodel_filename (str):  Filename of the external model
       data_dir (str):
       refmodel_filename (str, optional):
       csv_data (bool, optional):
       pandas_data (bool, optional):
       num_parameters (int, optional):
       multindex_cols_ext (list, optional):
       multindex_cols_ref (list, optional):
       features_name (str, optional):
       z_name (str, optional):
       features_to_Log (bool, optional):
       data_type= (str, optional):
       ratio_mode (bool, optional):
       param_names_dict (dict, optional):
       verbosity (int, optional):
       param_names_dict (dict, optional): dictionary containing parameter number as key and name of the parameter as value

    Attributes:
        ratio_mode (bool):
        self.flnm_ext (str):
        self.flnm_ref (str):
        self.data_dir (str):
        self.csv_bool (bool):
        self.pandas_bool (bool):
        self.num_parameters (int):
        self.mindexcols_ref (list):
        self.mindexcols_ext (list):
        self.features_str (str):
        self.z_str (str):
        self.data_type (str):
        self.features_to_Log (bool):
        self.paramnames_dict (dict):
    """


    def __init__(self, extmodel_filename,
                       data_dir,
                       refmodel_filename=None,
                       csv_data=True,
                       pandas_data=True,
                       num_parameters=1,
                       multindex_cols_ext=[0,1,2,3],
                       multindex_cols_ref=[0,1],
                       features_name='k_grid',
                       z_name='zred',
                       features_to_Log=True,
                       data_type='tcl',
                       ratio_mode=False,
                       param_names_dict={},
                       verbosity=1):

        self.ratio_mode = ratio_mode
        self.flnm_ext = extmodel_filename
        self.flnm_ref = refmodel_filename
        self.data_dir = data_dir
        self.csv_bool = csv_data
        self.pandas_bool = pandas_data
        self.num_parameters = num_parameters
        self.mindexcols_ref = multindex_cols_ref
        self.mindexcols_ext = multindex_cols_ref + list(range(2, 2 * (self.num_parameters + 1)))
        self.features_str = features_name
        self.z_str = z_name
        self.data_type = data_type
        self.features_to_Log=features_to_Log
        self.paramnames_dict = param_names_dict

        return None



    def read_csv_pandas(self, verbosity=1):
        """
        Read pandas dataframe from csv file

        Args:
            verbosity (int, optional): 
                determines whether or not details are printed (prints if verbosity >=3)
        """

        ## check if the input is a csv file
        if self.csv_bool==True:
            fileext = '.csv'
        else:
            raise ValueError('File type extension not supported yet.')
        
        ## read csv file
        self.df_ext = pd.read_csv(self.data_dir + self.flnm_ext + fileext, index_col=self.mindexcols_ext)

        ## get the feature grid (e.g. values of k) from the dataframe and read the reference dataframe
        if self.ratio_mode == False:
            self.df_ref = pd.read_csv(self.data_dir + self.flnm_ref + fileext, index_col=self.mindexcols_ref)
            self.fgrid = self.df_ref.loc[(self.features_str),:].values.flatten()
        else:
            self.fgrid  = self.df_ext.loc[(self.features_str),:].values.flatten()

        ## reverse log operation and set attribute as "lin_features_str" (e.g. "lin_kgrid")
        if self.features_to_Log==True:
            self.fgrid = np.log10(self.fgrid)
            setattr(self, 'lin_'+self.features_str, np.power(10,self.fgrid))

        ## set attribute for the feature grid as "features_str" (e.g. "kgrid")
        setattr(self, self.features_str, self.fgrid)

        ## if ratio_mode is not used, get multiindices of reference dataframe
        if self.ratio_mode ==False:
            too.condprint('Shape of imported reference model dataframe: ', str(self.df_ref.shape),level=3,verbosity=verbosity)
            self.multindex_ref = self.df_ref.index
            self.multindex_names_ref = list(self.multindex_ref.names)

        ## get multiindices from dataframe containing all spectra
        too.condprint('Shape of imported extended model dataframe: ', str(self.df_ext.shape),level=3,verbosity=verbosity)
        self.multindex_ext = self.df_ext.index
        self.multindex_names_ext = list(self.multindex_ext.names)

        ## get indexnames (strings) of redshift and data 
        z_indexname = [nn for nn in list(self.multindex_names_ext) if 'redshift' in nn][0]
        data_indexname = [nn for nn in list(self.multindex_names_ext) if 'data' in nn][0]
        
        ## get list of names of all data types in dataframe (e.g. ["tcl"]) 
        ## # TODO: Careful, this might break fot more than 1 param
        self.data_names = list(set([indi for indi in (self.multindex_ext.get_level_values(data_indexname)) if self.features_str not in indi]))
        self.data_names.sort()

        ## get list of all redshift values in the dataframe
        self.z_names = list(set([zst for zst in (self.multindex_ext.get_level_values(z_indexname).values) if np.isnan(zst)==False]))
        self.z_names.sort()
        
        ## get array of all redshift values as floats
        try:
            self.z_vals = [float((dd.replace(self.z_str+'_','')).replace('p','.')) for dd in self.z_names]
        except:
            self.z_vals = [float(dd) for dd in self.z_names]
        self.z_vals = np.array(self.z_vals)
        
        ## print multiindex if verbosity >= 3
        too.condprint("pandas DataFrame Multiindex", self.multindex_ext, level=3, verbosity=verbosity)
        
        ## sort dataframes 
        self.df_ext=self.df_ext.sort_values([data_indexname, z_indexname,'parameter_1_value'])
        if self.ratio_mode == False:
            self.df_ref=self.df_ref.sort_values([data_indexname, z_indexname])

        ## print values of parameter_1 if verbosity >=3
        too.condprint("pandas DataFrame Multiindex", self.multindex_ext.get_level_values('parameter_1_value').values, level=3, verbosity=verbosity)

        ## get list of parameter keys (e.g. ["parameter_1", "parameter_2", ...])
        Param_keys = list(self.df_ext.index.names)[2::2]

        ## get array of index values for all lines except the feature grid (e.g. array([["theo", "0.0", "omega_b", "0.02", ...], [...], ...]))
        Index = np.array([list(ii) for ii in self.df_ext.index.values if list(ii)[0] != 'k_grid' ])

        ## get array of parameter names (e.g. array(["omega_b", "omega_cdm", "H_0", ...]))
        Param_names = Index[0,2::2].flatten()

        ## get array of parameter values for all lines in dataframe (e.g array([[0.02, 0.12, 68.36, ...], [...], ...]))
        self.extparam_vals = np.unique(Index[:,3::2],axis=0).astype(np.float)

        ## get maximum size of training data set (whole data set -1 as we need at least 1 spectrum for testing)
        self.max_train = len(self.extparam_vals)-1
        
        ## create dictionary connecting parameter keys to parameter names (e.g. {"parameter_1": "omega_b", ...})
        try:
            param_names_dict ={}
            for i in range(len(Param_keys)):
                param_names_dict[str(Param_keys[i])]=str(Param_names [i])
            self.paramnames_dict = param_names_dict
        except:
            print("Parameters'names could not be read correctly. Please respect the standard format")


        return None



    def calculate_ratio_by_redshifts(self,redshift_list, normalize = True, pos_norm = 2):
        """Calculate the ratio between the external model and the reference for any redshift passed"

        Args:
            redshift_list (list): 
                redshifts desired for the ratios
            normalize (bool, optional):

            pos_norm (int, optional): 
                specify the position where the normalization takes the references. 
                At this point any ratio equals to 1
        """

        redshift_list = np.atleast_1d(redshift_list)
        
        ## check if there are spectra for multiple redshifts in the dataframe
        ## if multiple_z = False, the interpolator will construct an interpolation without taking the redshift into account
        if redshift_list.size==1:
            self.multiple_z = False
        else:
            self.multiple_z = True


        for z in np.sort(redshift_list):
            self.matrix_z = self.calculate_ratio_data(z,normalize,pos_norm, _SAVING=False)

        self.z_requested = np.array(redshift_list)



    def calculate_ratio_data(self, z, normalize = True, pos_norm = 2, _SAVING = True):
        """Calculate the ratio between the external model and the reference at a specified redshift

        Args:
            z (float): 
                redshift desired for the ratios
            normalize (bool, optional):
                if True, spectra are normalized by mean and standard deviation
            SAVING (bool, optional):  
                The user should not change this paremeter. True if several redshifts are computed, False otherwise.
        """

        try:
            z_digits = len((self.z_names[0].replace(self.z_str+'_','')).split('p')[1])  ##counts digits in a key like z_red_0p123456
            z_request = self.z_str+'_'+'{1:.{0}f}'.format(z_digits, z).replace('.','p')
        except:
            z_request = z
        if z_request not in self.z_names:  ## this allows to request numbers not exactly in z_vals if precision is still withing z_digits (usually 6)
            raise ValueError('Requested redshift'' is not contained in dataframe')



        ## if ratio mode is not used, the spectra are divided by a reference spectrum
        if self.ratio_mode == True:
            reftheo = 1
        else:
            reftheo = self.df_ref.loc[(self.data_type, z_request)].values

        exnoi = self.df_ext.loc[self.data_type, z_request].values / reftheo

        ## binwise normalization: we shift by the mean and divide by standard devitaion
        if normalize == True:
            binwise_mean = exnoi.mean(axis=0)
            binwise_std = exnoi.std(axis=0)
        else:
            binwise_mean = 0
            binwise_std = 1

        self.binwise_mean = binwise_mean
        self.binwise_std = binwise_std
        
        ## dictionary with data as keys and array of normalized spectra as values (e.g. {"theo": array([[...],[...],...])})
        matrix_z = (exnoi - binwise_mean) / binwise_std

        if _SAVING == True:
            self.matrix_z = matrix_z
            self.z_requested = z_request
            self.multiple_z = False
        else:
            return matrix_z



    @staticmethod
    def midpoint(arra):
        ll = len(arra)
        m = ll // 2
        return m


    @staticmethod
    def splitarra(arr):
        momo = DataHandle.midpoint(arr)
        lefto = arr[:momo + 1]
        righto = arr[momo:]
        return arr[momo], lefto, righto


    @staticmethod
    def left_right(arra, ind):
        mm, ll, rr = DataHandle.splitarra(arra)
        if ind % 2 == 0:
            return mm, rr
        elif ind % 2 == 1:
            return mm, ll


    @staticmethod
    def recurs(arra, index):
        mm, left, right = DataHandle.splitarra(arra)
        ss = index
        if ss == 1:
            return mm
        elif ss == 2:
            mi, ri = DataHandle.left_right(right, ss)
            return mi
        elif ss == 3:
            mi, le = DataHandle.left_right(left, ss)
            return mi
        while ss > 3:
            if index % 2 == 0:
                ss = ss // 2
                # print("ss even",ss)
                return DataHandle.recurs(right, ss)
            elif index % 2 == 1:
                ss = ss // 2
                # print("ss odd",ss)
                return DataHandle.recurs(left, ss)



    def data_separation(self, n_extrema=2, ind_extrema=[0,-1], verbosity=1):
        """Generates arrays of multi-index values representing subsets of the original data

        Args:
            n_extrema (int): 
                number of spectra considered "extrema"
            nind_extrema [list]:
                position index of these "extrema" (default: first and last spectra in dataframe)
            verbosity (int):
                determines whether or not details are printed, a higher value results in more output information
        """

        self.fullspace = []
        self.z_requested = np.array([self.z_requested]).flatten()

        ## fullspace: array of all multiindex values in the noiseless case 
        for z in np.array(self.z_requested):
            for iind in (self.df_ext.loc[self.data_type,z].index):
                ## values of the redshift are only included, if there is data for multiple redshift values
                if self.multiple_z == False:
                    self.fullspace.append(list(iind)[1::2])
                else:
                    self.fullspace.append(np.array([z]+list(iind)[1::2]))
        self.fullspace=np.array(self.fullspace)

        self.size_fullspace = len(self.fullspace) 
        self.ind_fullspace = np.array(range(self.size_fullspace))

        ## extremaspace: array of multiindex values of the spectra considered "extrema"
        self.ind_extremaspace = self.ind_fullspace[ind_extrema]
        self.extremaspace = self.fullspace[self.ind_extremaspace]
        self.size_extremaspace = len(self.extremaspace)

        ## midspace: array of all multiindex values except for the extrema
        self.ind_midspace = np.setdiff1d(self.ind_fullspace, self.ind_extremaspace)
        self.midspace = self.fullspace[self.ind_midspace]
        self.size_midspace = len(self.midspace)

        ## print details if verbosity >= level
        too.condprint("length of full sample space", self.size_fullspace, level=2, verbosity=verbosity)
        too.condprint("full sample space list", self.fullspace, level=3, verbosity=verbosity)
        too.condprint("length of extrema sample space", self.size_extremaspace, level=2, verbosity=verbosity)
        too.condprint("full sample space list", self.extremaspace, level=3, verbosity=verbosity)

        return None
    

    
    @staticmethod
    def stratify_array(array, num_percentiles=4):
        """
        This method takes in an array and a number of percentiles to divide the array into.
        It then stratifies the values in the array based on these percentiles and returns
        an array of labels indicating which percentile range each value belongs to.
    
        Parameters:
            array (numpy.ndarray): 
                The input array to be stratified.
            num_percentiles (int): 
                The number of percentiles to divide the array into. Default value is 4.
    
        Returns:
            stratif_labels (numpy.ndarray): 
                An array of labels indicating which percentile range each value belongs to.
        """

        percentiles = [ii*100/num_percentiles for ii in list(range(0,num_percentiles+1))]
        stratif_labels = np.copy(array)

        for ii in range(0,len(percentiles)-1):
            lo = percentiles[ii]
            hi = percentiles[ii+1]
            plo = np.percentile(array, lo)
            phi = np.percentile(array, hi)
            stratif_labels[( plo <=  array) & (array <= phi )] = ii+1

        return stratif_labels



    def calculate_data_split(self, n_train=2, n_vali=0, n_test=1, n_splits=1,
                             num_percentiles=4, random_state=87, verbosity=1,
                             manual_split=False,train_indices=None, test_indices=None,
                             train_redshift_indices=[0], test_redshift_indices=[0],
                             interpolate_over_redshift_only=False, **kwargs):
        """Calculate the splits indices of train,vali and test for each split. 

        Args:
            n_train (int):
                number of spectra in training data set
            n_vali (int):
                number of spectra in validation data set
            n_test (int):
                number of spectra in test data set
            n_split (int):
                number of different data splits
            num_percentiles:
            random_state:
            verbosity:
            manual_split:

            train_indices:
            test_indices:
            train_redshift_indices;
            test_redshift_indices:
            interpolate_over_redshift_only:
        """
        
        n_extrema=kwargs.get('n_extrema', 2)
        ind_extrema=kwargs.get('ind_extrema', [0,-1])
        self.data_separation(n_extrema=n_extrema, ind_extrema=ind_extrema)

        too.condprint("number of wanted training vectors", n_train, level=2, verbosity=verbosity)
        too.condprint("number of wanted test vectors", n_test, level=1, verbosity=verbosity)
        if n_train+n_test > (self.size_fullspace):
           print("Warning n_train is larger than total full sample space")

        self.random_state = random_state
        self.num_percentiles = num_percentiles
        self.n_splits = n_splits

        stratif_labels = self.stratify_array(self.midspace, num_percentiles=self.num_percentiles)
        self.test_splitdict = dict()
        self.train_splitdict = dict()
        self.vali_splitdict = dict()

        if manual_split == False:
            n_vali = self.size_midspace-n_test-n_train
            if n_vali !=0 and len(self.ind_midspace)> 1:
                kf = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=n_test, random_state=self.random_state)
                for ii, (trainvali, test) in enumerate(kf.split(self.midspace,stratif_labels)):
                    #test = test[np.in1d(test, extspace_ind, invert=True)]

                    test = self.ind_midspace[test]
                    if n_train > 0:
                        train, valitest = train_test_split(trainvali, test_size=n_vali, shuffle=True, random_state=self.random_state)
                        train = self.ind_midspace[train]
                        train  = np.unique(np.concatenate([train,self.ind_extremaspace]))
                        train = self.ind_fullspace[train]
                    else:
                        train  = self.ind_extremaspace
                        train = self.ind_fullspace[train]
                        valitest=trainvali

                    #valitest = valitest[np.in1d(valitest, extspace_ind, invert=True)]
                    valitest = self.ind_midspace[valitest]
                    #print(test, trr, " s tr", len(train)-2, " tr: ", train, " va: ", valitest)
                    self.test_splitdict[ii] = test
                    self.vali_splitdict[ii]= valitest
                    self.train_splitdict[ii] = train
            elif  len(self.ind_midspace)> 1 and n_vali == 0:
                kf = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=n_test, random_state=self.random_state)
                for ii, (train, test) in enumerate(kf.split(self.midspace,stratif_labels)):
                    test = self.ind_midspace[test]
                    train = self.ind_midspace[train]
                    train  = np.unique(np.concatenate([train,self.ind_extremaspace]))
                    train = self.ind_fullspace[train]
                    self.test_splitdict[ii] = test
                    self.train_splitdict[ii] = train

            else:
                 test = self.ind_midspace
                 train = self.ind_extremaspace
                 self.test_splitdict[0] = test
                 self.train_splitdict[0] = train
                 
        ###/!\ Warning /!\ For now we always use manual split (which not really manual now...)
        elif manual_split == True:
            ### Determine the number of samples avaible with different values of parameters. e.g nb_param = 101 for MassiveNus
            nb_param = int(len(self.fullspace)/len(self.z_requested)) 
            if len(self.z_requested)==1:
                nb_param = int(len(self.fullspace))
            

            for ii in range (n_splits):
                ###Here the user has chosen to provide the test indices
                if test_indices is not None:
                    test_indices = np.atleast_2d(test_indices)
                    test = test_indices[ii]
                    ###We make sure that the indice lies into a correct space. e.g if we have nb_param = 101, and a indices i = 103 it will become i =2
                    test_origin = [tt%nb_param for tt in test]
                     
                    ###Do we want to construct a interpolation only over the redshift ? /!\ Warning  /!\ this is case is not really used....
                    if interpolate_over_redshift_only == False and train_indices is None:
                        train_origin = [ii for ii in range(1,nb_param-1) if ii not in test_origin ]

                    elif  interpolate_over_redshift_only == False and train_indices is not None:
                         train_origin = [tt%nb_param for tt in train ]
                    else :
                        train_origin = test_origin
                ###Here the user has chosen not to provide the test indices
                ## so we first randomly generate them
                else:
                    if train_indices is None:
                       test_origin = [ii for ii in range(1,nb_param-1)]
                       test_origin =    shuffle(test_origin)[:n_test]
                       if interpolate_over_redshift_only == False:
                           train_origin = [ii for ii in range(1,nb_param-1) if ii not in test_origin ]
                       else:
                            train_origin = test_origin
                    ###The user has specified train indices so must be sure that train and test do not overlap !
                    else:
                        train_indices =  np.atleast_2d(train_indices)
                        train = train_indices[ii]
                        train_origin = [tt%nb_param for tt in train ]
                        test_origin = [ii for ii in range(1,nb_param-1) if ii not in train_origin ] ####!!!
                
                train_origin = shuffle(train_origin)
                
                train_origin = train_origin[:n_train]
                test_origin = shuffle(test_origin)[:n_test]
                if train_indices is None:
                    if [0] not in test_origin:
                        train_origin +=[0]
                    if [nb_param-1]not in  test_origin:
    
                        train_origin += [nb_param-1]
                    if [0] in test_origin or [nb_param-1] in test_origin :
                        print("Warning : trying to interpolate a extramal value")
                        

                train_redshift = self.z_requested[train_redshift_indices]
                test_redshift = self.z_requested[test_redshift_indices]
                self.train_redshift = train_redshift 
                self.test_redshift = test_redshift
                too.condprint("redshift used for training", train_redshift,level=1,verbosity=verbosity)
                too.condprint("redshfit used for testing", test_redshift,level=1,verbosity=verbosity)
                train  = []
                test  = []
                ### looping over the redshift 
                for zz in train_redshift_indices:
                    train+= [ii + zz*nb_param  for ii in train_origin  ]

                for zz in test_redshift_indices:    
                    test += [ii + zz*nb_param  for ii in test_origin  ]
                        

                self.train_splitdict[ii] = train
                self.test_splitdict[ii] = test
                shuffled = shuffle(train)
                self.train_splitdict[ii] = shuffled
                self.vali_splitdict[ii] = shuffled

        return None



    def data_split(self, split_index=0, thinning=None, apply_mask=False, mask=[], **kwargs):
        """Split the data into train, vali, test according to the indices calculated by calculate_data_split
        Args:
            split_index: index of split chosen 
            thinning: thinning of the grid
            apply_mask: mask applied or not
            mask: mask to apply
        Returns:
            self.matrix_datalearn_dict: Dictionary train,vali,test -> ratios
        """
        
        self.learn_sets = ['train','vali','test']
        self.ind_train = self.train_splitdict[split_index]
        self.ind_train.sort()
        self.ind_test = self.test_splitdict[split_index]
        self.ind_test.sort()
        if len(self.vali_splitdict) !=0:
            self.learn_sets = ['train','vali','test']
            self.ind_vali = self.vali_splitdict[split_index]
            self.ind_vali.sort()
            self.indices_learn_dict = dict(zip(self.learn_sets, [self.ind_train, self.ind_vali, self.ind_test]))
        else:
            self.learn_sets = ['train','test']
            self.indices_learn_dict = dict(zip(self.learn_sets, [self.ind_train, self.ind_test]))


        self.train_samples = self.fullspace[self.ind_train]
        self.train_size = len(self.train_samples)

        if len(self.vali_splitdict) !=0:
            self.vali_samples = self.fullspace[self.ind_vali]
            self.vali_samples.sort()
            self.vali_size = len(self.vali_samples)
        else:
            self.vali_size = 0
        self.test_samples = self.fullspace[self.ind_test]
        #self.test_samples.sort()
        self.test_size = len(self.test_samples)
        verbosity = kwargs.get('verbosity', 1)

        too.condprint("number of obtained training vectors", self.train_size, level=1, verbosity=verbosity)
        too.condprint("number of obtained validation vectors", self.vali_size, level=1, verbosity=verbosity)
        too.condprint("number of obtained test vectors", self.test_size, level=2, verbosity=verbosity)


        self.matrix_datalearn_dict = dict()

        for dli in self.learn_sets:
            matrixdata = np.copy(self.matrix_z)
            self.matrixdata=matrixdata

            ## copy of mask to avoid modifying orginal mask after iterations
            if apply_mask==False:
                maskcopy=np.arange(0,len(matrixdata[0])) ##range over all axis length, does not mask anything
            else:
                maskcopy=np.copy(mask)
            
            ## apply thinning (if set to None, there is no thinning)
            self.mask_true=maskcopy[::thinning] 

            ## apply mask also to feature grid and save as masked_+...
            setattr(self, 'masked_'+self.features_str, self.fgrid[self.mask_true]) 

            matrixdata = matrixdata[:,self.mask_true]  ## apply mask and thinning to feature space (k-grid)
            indices_l = self.indices_learn_dict[dli]
            matrixdata = matrixdata[indices_l,:]   ##choose learning set
            self.matrix_datalearn_dict[dli] = matrixdata
        self.matrix_datalearn_dict = objdict(self.matrix_datalearn_dict)
        return self.matrix_datalearn_dict



    @staticmethod
    def matrixdata_to_dict(data_mat, data_space):
        """Create a dictionary parameters -> spectra
        Args:
            data_space: space parameter
            data_mat: spectra
        Returns:
            dmatdata_dict: dictionary parameters -> spectra
        """
        
        matdata_dict = {}
        for vv,dd in zip(data_space, data_mat):
            matdata_dict[vv] = dd
        return matdata_dict





    def get_index_param(self, list_of_parameters_and_redshift, multiple_redshift=False):
        """
        Returns index which corresponds to a set of parameters

        Args:
            list_of_parameters_and_redshift: list of redshift (optional) and parameters 
            multiple_redshift: false if no redshift is provided
        Returns:
            ind: index where the parameters are located -> table.loc[ind]
        """
        idx = pd.IndexSlice
        if multiple_redshift:
            ind = idx[self.data_type,list_of_parameters_and_redshift[0]] # first value is the redshift
        else :
            ind = idx[self.data_type,:]
        for i in range (self.num_parameters):
            if multiple_redshift:
                ind += idx[:,list_of_parameters_and_redshift[i+1]] # first value is the redshift
            else : 
                ind += idx[:,list_of_parameters_and_redshift[i]] 
        return ind
            
