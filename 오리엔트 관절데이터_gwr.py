# -*- coding: utf-8 -*-
"""
gwr-tb :: Associative GWR demo
@last-modified: 30 November 2018
@author: German I. Parisi (german.parisi@gmail.com)
"""
import numpy as np
import gtls
from agwr import AssociativeGWR

# #정규분포(평균,시그마)
# rand_norm = np.random.normal(mu, sigma, size=(10,3))
# #t분포 (자유도,사이즈)
# rand_t = np.random.standard_t(df=3, size=(10,3))
# #균등분포 (최소,최대,사이즈)
# rand_unif = np.random.uniform(low=0.0, high=10.0, size=(10,3))
# #f분포 (자유도1,자유도2,사이즈)
# rand_f = np.random.f(dfnum=5, dfden=10, size=(10,3))
# #카이제곱분포 (자유도,사이즈)
# rand_chisq = np.random.chisquare(df=2, size=100)


if __name__ == "__main__":

    # Import dataset from file
    data_flag = True
    # Import pickled network
    import_flag = False
    # Train AGWR with imported dataset    
    train_flag = True
    # Compute classification accuracy    
    test_flag = False
    # Export pickled network     
    export_flag = False
    # Export result data
    result_flag = True
    # Plot network (2D projection)
    plot_flag = False
    
    if data_flag:
        ds_pose = gtls.bk_no_label_Dataset(file='전처리 csv\Part3(preprocessed)_nofloat.csv', normalize=True)
        print("%s from %s loaded." % (ds_pose.name, ds_pose.file))

    if import_flag:
        fname = 'my_agwr.agwr'
        my_net = gtls.import_network(fname, AssociativeGWR)

    if train_flag:
       # Create network 
       my_net = AssociativeGWR()
       # Initialize network with two neurons
       my_net.bk_no_lable_init_network(ds=ds_pose, random=False)
       # Train network on dataset
       my_net.bk_no_label_train(ds=ds_pose, epochs=10, a_threshold=0.3, l_rates=[0.2, 0.001],max_age=200, frame=1836, black=True)

    if test_flag:
        # Compute classification accuracy on given dataset
        my_net.test_agwr(ds_pose)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)
 
    if export_flag:
        fname = 'my_agwr.agwr'
        gtls.export_network(fname, my_net)

    if result_flag:
        fname = 'my_agwr_result.csv'
        gtls.export_result(fname, my_net, ds_pose)

    if plot_flag:
        gtls.plot_network(my_net, edges=True, labels=False)