import os

misc_folders = ["stored_models", "stored_models/figures"]

highlvl_folder = ['data']
midlvl_folders = ['lane-keep', 'lane-change-left', 'lane-change-right']
lowlvl_folders = ['0750am-0805am', '0805am-0820am','0820am-0835am',
                  '0400pm-0415pm', '0500pm-0515pm', '0515pm-0530pm']
folders = []
for misc in misc_folders:
    folders.append(misc)


for high in highlvl_folder:
    folders.append(high)
    for mid in midlvl_folders:
        folders.append(high + '/' + mid)
        for low in lowlvl_folders:
            folders.append(high + '/' + mid + '/' + low)

def setup_filestructure():
    for dir_name in folders:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            print("Directory " , dir_name ,  " Created ")
        else:    
            print("Directory " , dir_name ,  " already exists")
            
if __name__ == "__main__":
    setup_filestructure()