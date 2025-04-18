import os
import shutil

import datasets


WIDTH = shutil.get_terminal_size().columns
def load_and_print():
    print('TESTING 1/2 - CENTRALISED VERIONS'.center(WIDTH))
    centralised_dataset = datasets.load_from_disk(
        dataset_path=os.path.join(os.getcwd(), 'centralised', 'TISSUEMNIST')
    )
    print(centralised_dataset)
    print(('-'*WIDTH).center(WIDTH))
    print('TESTING 2/2 - DECENTRALISED VERIONS'.center(WIDTH))
    decentralised_dataset = datasets.load_from_disk(
        dataset_path=os.path.join(os.getcwd(), 'decentralised', 'TISSUEMNIST')
    )
    print(decentralised_dataset)
    print(('-'*WIDTH).center(WIDTH))
    print('TESTING COMPLETED'.center(WIDTH))


if __name__ == '__main__':
    load_and_print()