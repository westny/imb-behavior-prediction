import os 
from zipfile import ZipFile
from tqdm import tqdm


def unzip():
    files = os.listdir()
    try:
        f = [x for x in files if '.zip' in x][0]
    except IndexError:
        print('No .zip files found')
    else:
        val = input(f"Would you like to unzip \'{f}\' ? (Y|N) \n").lower()
        if 'y' in val or 'yes' in val:
            with ZipFile(f, 'r') as zf:
                for member in tqdm(zf.infolist(), desc='Extracting '):
                    try:
                        zf.extract(member)
                    except zipfile.error as e:
                        ...
            print('\n Done!')
        elif 'n' in val or 'no' in val:
            print('...')
        else:
            print('Unrecognized request')
            return False
    return True

unzip()