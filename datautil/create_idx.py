from subprocess import call
import os.path
import glob
from tqdm import tqdm
import os

tfrecord_path = '/home/paperspace/Dropbox/JQUAD/Data/waymo_motion/sample/'
idx_path = '/home/paperspace/Dropbox/JQUAD/Data/waymo_motion/sample_idx/'

os.makedirs(idx_path, exist_ok=True)

for tfrecord in tqdm(glob.glob(tfrecord_path+'/*')):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    call(["tfrecord2idx", tfrecord, idxname])
