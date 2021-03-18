import os
from random import seed
import src.lib.network as net
import src.lib.image_utils as imgt
import src.lib.utils as ut
import src.lib.enum_utils as eut

def main():
    #BASE-TEST with the original dataset for images
    train_dir_base_path = "./src/resources/imgs/train"
    test_dir_base_path = "./src/resources/imgs/test"

    dataset = imgt.feature_extraction_from_dir(train_dir_base_path)
    #imgt.print_feature_matrix(dataset)
    print("[BASE-TEST] after feature matrix created")

    n_inputs = len(dataset[0]) - 1
    n_hidden = int(len(dataset) / 2) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    l_rate = 0.5
    epochs = 200
    seed(1)
    print("n_inputs: %f\tn_hidden: %f\tn_outputs: %f" % (n_inputs, n_hidden, n_outputs))
    
    network = net.initialize_network(n_inputs, n_hidden, n_outputs)
    net.train_network(network, dataset, l_rate, epochs, n_outputs, eut.TraningFunctions.SIGMOID)
    #net.print_network(network)
    print("net trained")

    testset = imgt.feature_extraction_from_dir(test_dir_base_path)
    #imgt.print_feature_matrix(testset)
    print("\n[BASE-TEST] after test feature matrix created")
    for row in testset:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))

    #DOCT-TEST with doctored images doctored again (hinstogram alterated)
    train_dir_base_path = "./src/resources/supervised_doct/train"
    test_dir_base_path = "./src/resources/supervised_doct/test"
    test_unsupervised_dir_base_path = "./src/resources/unsupervised/test"

    dataset = imgt.feature_extraction_from_dir(train_dir_base_path)
    #imgt.print_feature_matrix(dataset)
    print("\n[DOCT-TEST] after feature matrix created")

    n_inputs = len(dataset[0]) - 1
    n_hidden = int(len(dataset) / 2) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    l_rate = 0.5
    epochs = 200
    seed(1)
    print("n_inputs: %f\tn_hidden: %f\tn_outputs: %f" % (n_inputs, n_hidden, n_outputs))
    
    network = net.initialize_network(n_inputs, n_hidden, n_outputs)
    net.train_network(network, dataset, l_rate, epochs, n_outputs, eut.TraningFunctions.SIGMOID)
    #net.print_network(network)
    print("net trained")

    testset = imgt.feature_extraction_from_dir(test_dir_base_path)
    #imgt.print_feature_matrix(testset)
    print("\n[DOCT-TEST] after test feature matrix created")
    for row in testset:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))


    testset_uns = imgt.feature_extraction_from_dir_unsupervised(test_unsupervised_dir_base_path)    
    print("[DOCT-TEST] after testset_uns feature matrix created")
    for row in testset_uns:
        prediction = net.predict(network, row, eut.TraningFunctions.SIGMOID)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))

if __name__ == "__main__":
    main()