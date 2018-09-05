# coding: utf-8
import pickle
import shutil
import os
import paramiko
from scp import SCPClient
from config import Config

opt = Config()

#创建ssh访问
port = 22
hostname = '115.156.207.244'
username = 'dian'
password = 'DianSince2002'
local_path = './source/bad_case/%s_bad_case.pkl'%opt.DATASET_PATH
remote_path = '~/miracle/auto-wash/source/bad_case/%s_bad_case.pkl'%opt.DATASET_PATH

def createSSHClient(hostname, port, username, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname, port, username, password)
    return client

ssh = createSSHClient(hostname, port, username, password)
scp = SCPClient(ssh.get_transport())
scp.get(local_path=local_path, remote_path=remote_path)

bad_case_names = pickle.load(open(local_path, 'rb'))

dataset_path = './source/bad_case_images/%s'%opt.DATASET_PATH
if not os.path.exists(dataset_path): os.mkdir(dataset_path)
folders = list(set([x.split('/')[-2] for x in bad_case_names]))
for folder in folders:
    temp_path = './source/bad_case_images/'+ opt.DATASET_PATH + '/' + folder
    if not os.path.exists(temp_path): os.mkdir(temp_path)

for path in bad_case_names:
    new_path = './source/bad_case_images/' + opt.DATASET_PATH + '/' + '/'.join(path.split('/')[-2:])
    shutil.copyfile(path, new_path)
    # os.remove(bad_case_names)
    print("%s has been removed."%path)
print("==> All bad cases have been removed.")