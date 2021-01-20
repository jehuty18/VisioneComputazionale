import os
from random import seed
import src.lib.network as net
import src.lib.image_utils as imgt
import src.lib.utils as ut

def main():
    dir = "./src/resources/imgs"

    dataset = imgt.feature_extraction_from_dir(dir)
    imgt.print_feature_matrix(dataset)

    n_inputs = len(dataset[0]) - 1
    n_hidden = int(len(dataset) / 2) - 1
    n_outputs = len(set([row[-1] for row in dataset]))
    l_rate = 0.01
    epochs = 500

    seed(1)
    network = net.initialize_network(n_inputs, n_hidden, n_outputs)
    net.train_network(network, dataset, l_rate, epochs, n_outputs)
    #net.print_network(network)

    ut.save_matrix_bin(network, "network")

    """for row in testset:
        prediction = net.predict(network, row)
        if row[-1] is None:
            print('Blind evaluation, Got=%d' % (prediction))
        else:	
            print('Expected=%d, Got=%d' % (row[-1], prediction))"""

if __name__ == "__main__":
    main()