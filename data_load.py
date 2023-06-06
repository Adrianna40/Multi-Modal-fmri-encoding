from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os 
import zipfile
import numpy as np



class SubjPaths:
  def __init__(self, data_dir, subj):
    
    self.subj = format(subj, '02')
    self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
    self.img_dir = os.path.join(self.data_dir, 'training_split', 'training_images')
    self.lh_fmri_path = os.path.join(self.data_dir, 'training_split', 'training_fmri', 'lh_training_fmri.npy')
    self.rh_fmri_path = os.path.join(self.data_dir, 'training_split', 'training_fmri', 'rh_training_fmri.npy')

    self.img_dir_test = os.path.join(self.data_dir, 'test_split', 'test_images')

    self.roi_masks = os.path.join(self.data_dir, 'roi_masks')
    self.img_list = os.listdir(self.img_dir)
    self.img_list.sort()
    self.img_dir_list = [os.path.join(self.img_dir, img) for img in self.img_list]

    self.img_list_test = os.listdir(self.img_dir_test)
    self.img_list_test.sort()
    self.img_dir_list_test = [os.path.join(self.img_dir_test, img) for img in self.img_list_test]
    self.parent_submission_dir = 'submission'
    self.subject_submission_dir = os.path.join(self.parent_submission_dir,
        'subj'+self.subj)

    # Create the submission directory if not existing
    if not os.path.isdir(self.subject_submission_dir):
        os.makedirs(self.subject_submission_dir)

class SubjData:
    def __init__(self, subj_paths):
        self.lh_fmri = np.load(subj_paths.lh_fmri_path)
        self.rh_fmri = np.load(subj_paths.rh_fmri_path)
   
        print('LH training fMRI data shape:')
        print(self.lh_fmri.shape)
        print('(Training stimulus images × LH vertices)')

        print('\nRH training fMRI data shape:')
        print(self.rh_fmri.shape)
        print('(Training stimulus images × RH vertices)')


def get_file_id(parent_id, file_name, drive):
    fileList = drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
    for file in fileList:
        if file['title'] == file_name:
            print('found dataset folder')
            return file['id']
    
def get_file(parent_id, file_name, drive):
    file_list = drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
    for file in file_list:
        if file['title'] == file_name:
            print('getting content of the file')
            file.GetContentFile(os.path.join('algonauts_2023_challenge_data', file_name))
            return 


def get_subj_dataset(subj, dataset_dir):
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    dataset_id = get_file_id('root', 'algonauts', drive)
    get_file(dataset_id, f'subj0{subj}.zip', drive)
    with zipfile.ZipFile(f'{dataset_dir}/subj0{subj}', 'r') as zip_ref:
        zip_ref.extractall(f'{dataset_dir}/subj0{subj}')
    subj_paths = SubjPaths(dataset_dir, subj)
    subj_data = SubjData(subj_paths)
    return subj_paths, subj_data



