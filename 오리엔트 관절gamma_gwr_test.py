import matplotlib.pyplot as plt
import gtls
from NoLabel_gamma_gwr import NoLabelGammaGWR

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
    export_flag = True
     # Export result data
    result_flag = True
    # Plot network (2D projection)
    plot_flag = True

    if data_flag:
        ds_gamma = gtls.bk_no_label_Dataset(
            file= 'C:\\Users\\zjkee\\OneDrive\\바탕 화면\\rpy\\video_cut\\4_front_2\\1\\1.mp4_pose_world_face(preprocessed).csv', normalize=True)
        print("%s from %s loaded." % (ds_gamma.name, ds_gamma.file))

    if import_flag:
        fname = 'my_net.ggwr'
        my_net = gtls.import_network(fname, NoLabelGammaGWR)

    if train_flag:
        # Create network
        my_net = NoLabelGammaGWR()
        # Initialize network with two neurons
        my_net.init_network(ds=ds_gamma, random=False, num_context=5)

        my_net.train_gammagwr(ds=ds_gamma, epochs=30,
                                a_threshold=0.225, max_age=3080, beta=0.7, l_rates=[0.2, 0.001])


    if test_flag:
        my_net.test_gammagwr(ds_gamma, test_accuracy=True)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)

    if export_flag:
        fname = 'my_net_ggwr'
        gtls.export_network(fname, my_net)

    if result_flag:
        fname = 'C:\\Users\\zjkee\\OneDrive\\바탕 화면\\rpy\\video_cut\\4_front_2\\1\\1.mp4_pose_world_face(preprocessed)_gammagwr(0.15,2942).csv'
        gtls.export_result(fname, my_net, ds_gamma)

    if plot_flag:
        gtls.plot_network(my_net, edges=True, labels=False)


