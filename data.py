from utils import update_dataset


for ds in ['live', 'train', 'validation']:
    update_dataset(ds)