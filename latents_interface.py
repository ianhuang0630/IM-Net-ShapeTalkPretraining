from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse

import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from modelAE import IM_AE, im_network

import pandas as pd
from tqdm import tqdm
import mcubes

import numpy as np
import h5py
import torch

from six.moves import cPickle
import math

from PIL import Image

def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data().
    :param file_name: file holding the pickled data.
    :param python2_to_3: (boolean), if True, pickle happened under python2x, unpickling under python3x.
    :return: an generator over the un-pickled items.
    Note, about implementing the python2_to_3 see
        https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
    """
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()


def pickle_data(file_name, *args):
    """Using (c)Pickle to save multiple python objects in a single file.
    """
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


class IM_AE_inference(object):
    def __init__(self, config):
        #progressive training
        #1-- (16, 16*16*16)
        #2-- (32, 16*16*16)
        #3-- (64, 16*16*16*4)
        self.sample_vox_size = config.sample_vox_size
        if self.sample_vox_size==16:
            self.load_point_batch_size = 16*16*16
            self.point_batch_size = 16*16*16
            self.shape_batch_size = 32
        elif self.sample_vox_size==32:
            self.load_point_batch_size = 16*16*16
            self.point_batch_size = 16*16*16
            self.shape_batch_size = 32
        elif self.sample_vox_size==64:
            self.load_point_batch_size = 16*16*16*4
            self.point_batch_size = 16*16*16
            self.shape_batch_size = 32
        self.input_size = 64 #input voxel grid size

        self.ef_dim = 32
        self.gf_dim = 128
        self.z_dim = 256
        self.point_dim = 3

        
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir
        self.dataset_name = config.dataset

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #build model
        self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.im_network.to(self.device)
        #print params
        #for param_tensor in self.im_network.state_dict():
        #    print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
        self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
        #pytorch does not have a checkpoint manager
        #have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name='IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0

        #keep everything a power of 2
        self.cell_grid_size = 4
        self.frame_grid_size = 64
        self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
        self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
        self.test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change

        #get coords for training
        dima = self.test_size
        dim = self.frame_grid_size
        self.aux_x = np.zeros([dima,dima,dima],np.uint8)
        self.aux_y = np.zeros([dima,dima,dima],np.uint8)
        self.aux_z = np.zeros([dima,dima,dima],np.uint8)
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier
        multiplier3 = multiplier*multiplier*multiplier
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i,j,k] = i*multiplier
                    self.aux_y[i,j,k] = j*multiplier
                    self.aux_z[i,j,k] = k*multiplier
        self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
        self.coords = (self.coords.astype(np.float32)+0.5)/dim-0.5
        self.coords = np.reshape(self.coords,[multiplier3,self.test_point_batch_size,3])
        self.coords = torch.from_numpy(self.coords)
        self.coords = self.coords.to(self.device)
        

        #get coords for testing
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc,dimc,dimc],np.int32)
        self.cell_y = np.zeros([dimc,dimc,dimc],np.int32)
        self.cell_z = np.zeros([dimc,dimc,dimc],np.int32)
        self.cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
        self.frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
        self.frame_x = np.zeros([dimf,dimf,dimf],np.int32)
        self.frame_y = np.zeros([dimf,dimf,dimf],np.int32)
        self.frame_z = np.zeros([dimf,dimf,dimf],np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i,j,k] = i
                    self.cell_y[i,j,k] = j
                    self.cell_z[i,j,k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i,j,k,:,:,:,0] = self.cell_x+i*dimc
                    self.cell_coords[i,j,k,:,:,:,1] = self.cell_y+j*dimc
                    self.cell_coords[i,j,k,:,:,:,2] = self.cell_z+k*dimc
                    self.frame_coords[i,j,k,0] = i
                    self.frame_coords[i,j,k,1] = j
                    self.frame_coords[i,j,k,2] = k
                    self.frame_x[i,j,k] = i
                    self.frame_y[i,j,k] = j
                    self.frame_z[i,j,k] = k
        self.cell_coords = (self.cell_coords.astype(np.float32)+0.5)/self.real_size-0.5
        self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
        self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
        self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
        self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
        self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
        self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
        self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
        self.frame_coords = (self.frame_coords.astype(np.float32)+0.5)/dimf-0.5
        self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])
        
        self.sampling_threshold = 0.5 #final marching cubes threshold
        # self.sampling_threshold = 0.3

    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.dataset_name, self.input_size)

    
    def z2voxel(self, z):
        model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        
        frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
        queue = []
        
        frame_batch_num = int(dimf**3/self.test_point_batch_size)
        assert frame_batch_num>0
        
        #get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)
            _, model_out_ = self.im_network(None, z, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            y_coords = self.frame_y[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            z_coords = self.frame_z[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])
        
        #get queue and fill up ones
        for i in range(1,dimf+1):
            for j in range(1,dimf+1):
                for k in range(1,dimf+1):
                    maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
                    minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
                    if maxv!=minv:
                        queue.append((i,j,k))
                    elif maxv==1:
                        x_coords = self.cell_x+(i-1)*dimc
                        y_coords = self.cell_y+(j-1)*dimc
                        z_coords = self.cell_z+(k-1)*dimc
                        model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
        
        print("running queue:",len(queue))
        cell_batch_size = dimc**3
        cell_batch_num = int(self.test_point_batch_size/cell_batch_size)
        assert cell_batch_num>0
        #run queue
        while len(queue)>0:
            batch_num = min(len(queue),cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0]-1,point[1]-1,point[2]-1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            _, model_out_batch_ = self.im_network(None, z, cell_coords, is_training=False)
            model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
                x_coords = self.cell_x+(point[0]-1)*dimc
                y_coords = self.cell_y+(point[1]-1)*dimc
                z_coords = self.cell_z+(point[2]-1)*dimc
                model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
                
                if np.max(model_out)>self.sampling_threshold:
                    for i in range(-1,2):
                        pi = point[0]+i
                        if pi<=0 or pi>dimf: continue
                        for j in range(-1,2):
                            pj = point[1]+j
                            if pj<=0 or pj>dimf: continue
                            for k in range(-1,2):
                                pk = point[2]+k
                                if pk<=0 or pk>dimf: continue
                                if (frame_flag[pi,pj,pk] == 0):
                                    frame_flag[pi,pj,pk] = 1
                                    queue.append((pi,pj,pk))
        return model_float

    def get_data_new(self, voxel_input_dir, splits_csv, mode):
        vox_samplings_top_dir = voxel_input_dir
        traintest_splits = splits_csv 
        classes = os.listdir(vox_samplings_top_dir)

        all_voxels = []
        all_pt = []
        all_val = []
        name_list = [] 

        for class_idx, class_name in enumerate(classes):
            train_test_splits_csv = pd.read_csv(traintest_splits)
            subsplit = train_test_splits_csv[train_test_splits_csv['shape_class'] == class_name].to_dict(orient='records')
            dataset_id_to_traintest = {} 
            for tup in subsplit:
                dataset_id_to_traintest[(tup['dataset'], tup['model_name'])] = tup['split']

            input_txt_dir = os.path.join(vox_samplings_top_dir, class_name, class_name+'_vox256.txt')
            input_txt = open(input_txt_dir, 'r')
            # this list is already sorted
            input_list = [el.strip() for el in input_txt.readlines()]
            input_txt.close()

            shape_name_list = []
            shape_name_idx = []
            for idx, inp in enumerate(input_list):
                # check  which split  it belongs  to
                path_elements = inp.split('/')
                dataset_name = path_elements[0]
                model_id = path_elements[-1]
                if (dataset_name, model_id) not in dataset_id_to_traintest:
                    continue
                split_type = dataset_id_to_traintest[(dataset_name, model_id)]
                if split_type == mode:
                    shape_name_list.append(class_name + '/' + inp)
                    shape_name_idx.append(idx)

            print('class {} : {}    num_objects: {}'.format(class_idx + 1, class_name, len(shape_name_idx)))

            voxel_hdf5_dir1 = os.path.join(vox_samplings_top_dir, class_name, class_name+'_vox256.hdf5')
            voxel_hdf5_file1 = h5py.File(voxel_hdf5_dir1, 'r')
            voxel_hdf5_voxels = voxel_hdf5_file1['voxels'][:][shape_name_idx]
            voxel_hdf5_points = (voxel_hdf5_file1['points_{}'.format(self.sample_vox_size)][:][shape_name_idx].astype(np.float32)+0.5)/256-0.5
            voxel_hdf5_values = voxel_hdf5_file1['values_{}'.format(self.sample_vox_size)][:][shape_name_idx].astype(np.float32)
            voxel_hdf5_file1.close()

            name_list.extend(shape_name_list)    
            all_voxels.append(voxel_hdf5_voxels)
            all_pt.append(voxel_hdf5_points)
            all_val.append(voxel_hdf5_values)

        all_voxels = np.concatenate(all_voxels)    
        all_pt = np.concatenate(all_pt)
        all_val = np.concatenate(all_val)
        all_voxels = np.reshape(all_voxels, [-1,1,self.input_size,self.input_size,self.input_size])

        return name_list, all_voxels, all_pt, all_val

    def get_dataset(self, set_name):
        assert set_name in ('train', 'test', 'val')
        dataset_load = self.dataset_name + '_{}'.format(set_name)
        # if not (config.train or config.getz):
        data_hdf5_name = self.data_dir+'/'+dataset_load+'.hdf5'

        id_hdf5_name = self.data_dir+'/'+dataset_load+'.txt'
        with open(id_hdf5_name, 'r') as f:
            shape_ids = [el.split() for el in f.readlines()]

        if os.path.exists(data_hdf5_name):
            data_dict = h5py.File(data_hdf5_name, 'r')
            data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
            data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
            data_voxels = data_dict['voxels'][:]
            #reshape to NCHW
            data_voxels = np.reshape(data_voxels, [-1,1,self.input_size,self.input_size,self.input_size])
        else:
            print("error: cannot load "+data_hdf5_name)
            exit(0)
        return shape_ids, data_voxels, data_points, data_values

class ImNetWrapper(object):
    def __init__(self, config):
        self.config = config
        self.im_ae = IM_AE_inference(config)
        checkpoint_txt = os.path.join(self.im_ae.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_ae.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return
    # somehow this doesn't get pickled if it's imported.
    def write_ply_triangle(self, name, vertices, triangles):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("element face "+str(len(triangles))+"\n")
        fout.write("property list uchar int vertex_index\n")
        fout.write("end_header\n")
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
        for ii in range(len(triangles)):
            fout.write("3 "+str(triangles[ii,0])+" "+str(triangles[ii,1])+" "+str(triangles[ii,2])+"\n")
        fout.close()
    
    def write_ply_point_normal(self, name, vertices, normals=None):
        fout = open(name, 'w')
        fout.write("ply\n")
        fout.write("format ascii 1.0\n")
        fout.write("element vertex "+str(len(vertices))+"\n")
        fout.write("property float x\n")
        fout.write("property float y\n")
        fout.write("property float z\n")
        fout.write("property float nx\n")
        fout.write("property float ny\n")
        fout.write("property float nz\n")
        fout.write("end_header\n")
        if normals is None:
            for ii in range(len(vertices)):
                fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(vertices[ii,3])+" "+str(vertices[ii,4])+" "+str(vertices[ii,5])+"\n")
        else:
            for ii in range(len(vertices)):
                fout.write(str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+" "+str(normals[ii,0])+" "+str(normals[ii,1])+" "+str(normals[ii,2])+"\n")
        fout.close()


    
    def sample_points_triangle(self, vertices, triangles, num_of_points):
        epsilon = 1e-6
        triangle_area_list = np.zeros([len(triangles)],np.float32)
        triangle_normal_list = np.zeros([len(triangles),3],np.float32)
        for i in range(len(triangles)):
            #area = |u x v|/2 = |u||v|sin(uv)/2
            a,b,c = vertices[triangles[i,1]]-vertices[triangles[i,0]]
            x,y,z = vertices[triangles[i,2]]-vertices[triangles[i,0]]
            ti = b*z-c*y
            tj = c*x-a*z
            tk = a*y-b*x
            area2 = math.sqrt(ti*ti+tj*tj+tk*tk)
            if area2<epsilon:
                triangle_area_list[i] = 0
                triangle_normal_list[i,0] = 0
                triangle_normal_list[i,1] = 0
                triangle_normal_list[i,2] = 0
            else:
                triangle_area_list[i] = area2
                triangle_normal_list[i,0] = ti/area2
                triangle_normal_list[i,1] = tj/area2
                triangle_normal_list[i,2] = tk/area2
        
        triangle_area_sum = np.sum(triangle_area_list)
        sample_prob_list = (num_of_points/triangle_area_sum)*triangle_area_list

        triangle_index_list = np.arange(len(triangles))

        point_normal_list = np.zeros([num_of_points,6],np.float32)
        count = 0
        watchdog = 0

        while(count<num_of_points):
            np.random.shuffle(triangle_index_list)
            watchdog += 1
            if watchdog>100:
                print("infinite loop here!")
                return point_normal_list
            for i in range(len(triangle_index_list)):
                if count>=num_of_points: break
                dxb = triangle_index_list[i]
                prob = sample_prob_list[dxb]
                prob_i = int(prob)
                prob_f = prob-prob_i
                if np.random.random()<prob_f:
                    prob_i += 1
                normal_direction = triangle_normal_list[dxb]
                u = vertices[triangles[dxb,1]]-vertices[triangles[dxb,0]]
                v = vertices[triangles[dxb,2]]-vertices[triangles[dxb,0]]
                base = vertices[triangles[dxb,0]]
                for j in range(prob_i):
                    #sample a point here:
                    u_x = np.random.random()
                    v_y = np.random.random()
                    if u_x+v_y>=1:
                        u_x = 1-u_x
                        v_y = 1-v_y
                    ppp = u*u_x+v*v_y+base
                    
                    point_normal_list[count,:3] = ppp
                    point_normal_list[count,3:] = normal_direction
                    count += 1
                    if count>=num_of_points: break

        return point_normal_list

    def get_z(self, voxel_input_dir, splits_csv, dsets=None):
        if dsets is None:
            dataset_types = ('train', 'val', 'test')
        else:
            if isinstance(dsets, tuple):
                dataset_types = dsets
            else:
                dataset_types = set([dsets])
        final_ids = [] # [None]*len(dataset_types)
        final_zs = [] # [None]*len(dataset_types)
        for i, dataset_type in enumerate(dataset_types): 

            this_ids, data_voxels, _, _ = self.im_ae.get_data_new(voxel_input_dir, splits_csv, dataset_type)
            final_ids.extend(this_ids)
            shape_num = len(data_voxels)
            hdf5_path = self.im_ae.checkpoint_dir+'/'+self.im_ae.model_dir+'/'+self.im_ae.dataset_name+'_{}_z.hdf5'.format(dataset_type) # config.z_postfix #'_train_z.hdf5'
            hdf5_file = h5py.File(hdf5_path, mode='w')
            hdf5_file.create_dataset("zs", [shape_num,self.im_ae.z_dim], np.float32)

            self.im_ae.im_network.eval()
            print(shape_num)
            for t in tqdm(range(shape_num)):
                batch_voxels = data_voxels[t:t+1].astype(np.float32)
                batch_voxels = torch.from_numpy(batch_voxels)
                batch_voxels = batch_voxels.to(self.im_ae.device)
                out_z,_ = self.im_ae.im_network(batch_voxels, None, None, is_training=False)
                hdf5_file["zs"][t:t+1,:] = out_z.detach().cpu().numpy()
            hdf5_file.close()
            hdf5_file = h5py.File(hdf5_path, mode='r')
            for vec in hdf5_file['zs'][:]:
                final_zs.append(vec)
            hdf5_file.close()

        return dict(zip(final_ids, final_zs))
    
    @torch.no_grad()
    def eval_z(self, z, optimize_mesh=False, compute_pc=True, npc_points=4096, save_output=False, 
               skip_existing=False, output_dir=None, verbose=False, return_results=True, shuffle_order=False, iso_value=None):
        ids = []
        zs = [] 
        for id_ in sorted(list(z.keys()), reverse=True):
            ids.append(id_)
            zs.append(z[id_])        
        zs = np.stack(zs)
        
        if shuffle_order:
            ids = np.array(ids)
            ridx = np.arange(len(ids))
            np.random.shuffle(ridx)
            ids = ids[ridx]
            zs = zs[ridx]
            
        self.im_ae.im_network.eval()
        meshes = dict()
        pcs = dict()
                
        if iso_value is None:
            iso_value = self.im_ae.sampling_threshold
            
        for i in tqdm(range(len(zs))):
            if verbose:
                print(ids[i])
            
            if skip_existing and output_dir is not None:
                mesh_path = output_dir + "/" + ids[i] + ".ply"
                if os.path.exists(mesh_path):
                    continue

            model_z = torch.tensor(zs[i:i+1]).to(self.im_ae.device)
            model_float = self.im_ae.z2voxel(model_z)
            vertices, triangles = mcubes.marching_cubes(model_float, iso_value)
            vertices = (vertices.astype(np.float32) - 0.5) / self.im_ae.real_size - 0.5
            
            if optimize_mesh:
                vertices = self.im_ae.optimize_mesh(vertices,model_z)

            if return_results:
                meshes[ids[i]] = (vertices, triangles)
                        
            if save_output:                
                mesh_path = output_dir + "/" + ids[i] + ".ply"
                if not os.path.exists(os.path.dirname(mesh_path)):
                    os.makedirs(os.path.dirname(mesh_path))
                self.write_ply_triangle(mesh_path, vertices, triangles)
                        
            ##  sample surface points
            if compute_pc:
                sampled_points_normals = self.sample_points_triangle(vertices, triangles, npc_points)
                final_pc = sampled_points_normals[:, :3].squeeze()
                if return_results:
                    pcs[ids[i]] = final_pc
                
                if save_output:
                    pc_path = output_dir + "/" + ids[i] + '.npz'
                    if not os.path.exists(os.path.dirname(pc_path)):
                        os.makedirs(os.path.dirname(pc_path))                
                    np.savez(pc_path, pc=final_pc)                        
                
        return meshes, pcs


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--svr", action="store_true", dest="svr", default=False, help="True for svr [False]")
parser.add_argument("--getz", action="store_true", dest="getz", default=False, help="True for getting latent codes [False]")
parser.add_argument("--z_postfix", default='_train_z.hdf5', type=str)
FLAGS = parser.parse_args()

imw = ImNetWrapper(FLAGS)
print(imw)

import dill as pickle

with open('IMNET-latent-interface-ld3de-pub.pkl', 'wb') as f:
    pickle.dump(imw, f)

print('Dilled the interface at IMNET-latent-interface-ld3de-pub.pkl')