import matplotlib.pyplot as plt
import gtls
from NoLabel_gamma_gwr import NoLabelGammaGWR
import optuna

# Define the objective function for Optuna
def objective(trial):
    # Define hyperparameter search spaces
    epochs = trial.suggest_int('epochs', 10, 50)
    a_threshold = trial.suggest_float('a_threshold', 0.01, 0.5)
    max_age = trial.suggest_int('max_age', 100, 3000)
    beta = trial.suggest_float('beta', 0.1, 1.0)
    l_rate_1 = trial.suggest_float('l_rate_1', 0.01, 0.5)
    l_rate_2 = trial.suggest_float('l_rate_2', 0.0001, 0.01)

    # Create network
    my_net = NoLabelGammaGWR()
    # Initialize network with two neurons
    my_net.init_network(ds=ds_gamma, random=False, num_context=1)

    # Train with the suggested hyperparameters
    my_net.train_gammagwr(ds=ds_gamma, epochs=epochs, a_threshold=a_threshold,
                          max_age=max_age, beta=beta, l_rates=[l_rate_1, l_rate_2])

    # Return the negative accuracy as Optuna tries to minimize the objective
    return -my_net.test_accuracy

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
            file='C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨_첫행feature없음.csv', normalize=True)
        print("%s from %s loaded." % (ds_gamma.name, ds_gamma.file))

    if import_flag:
        fname = 'my_net.ggwr'
        my_net = gtls.import_network(fname, NoLabelGammaGWR)

    if train_flag:
        # Create a study object and optimize the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)  # You can increase n_trials for a more extensive search

        # Get the best hyperparameters
        best_params = study.best_params
        print("Best Hyperparameters:", best_params)

        # Create network with the best hyperparameters
        my_net = NoLabelGammaGWR()
        my_net.init_network(ds=ds_gamma, random=False, num_context=1)
        my_net.train_gammagwr(ds=ds_gamma, epochs=best_params['epochs'],
                              a_threshold=best_params['a_threshold'],
                              max_age=best_params['max_age'],
                              beta=best_params['beta'],
                              l_rates=[best_params['l_rate_1'], best_params['l_rate_2']])

    if test_flag:
        my_net.test_gammagwr(ds_gamma, test_accuracy=True)
        print("Accuracy on test-set: %s" % my_net.test_accuracy)

    if export_flag:
        fname = 'my_net_ggwr'
        gtls.export_network(fname, my_net)

    if result_flag:
        fname = 'C:\\Users\\hslee\\Desktop\\dataset\\HYEONSU\\PROCESS4_FRONT_PART1_CSV\\결과\\Part1.mp4pose_world_visibility사라짐_하반신제거_결측치처리됨_첫행feature없음_결과.csv'
        gtls.export_result(fname, my_net, ds_gamma)

    if plot_flag:
        gtls.plot_network(my_net, edges=True, labels=False)
