import glob
import os
import numpy as np
from psbody.mesh import Mesh, MeshViewer, MeshViewers
import time
from copy import deepcopy
import random
from sklearn.decomposition import PCA
from tqdm import tqdm

class FaceData(object):
	def __init__(self, nVal, train_file, test_file, reference_mesh_file,pca_n_comp=8):
		self.nVal = nVal
		self.train_file = train_file
		self.test_file = test_file
		self.vertices_train = None
		self.vertices_val = None
		self.vertices_test = None
		self.N = None
		self.n_vertex = None

		self.load()
		self.reference_mesh = Mesh(filename=reference_mesh_file)

		self.mean = np.mean(self.vertices_train, axis=0)
		self.std = np.std(self.vertices_train, axis=0)
		self.pca = PCA(n_components=pca_n_comp)
		self.pcaMatrix = None
		self.normalize()

	def load(self):
		self.vertices_train = np.load(self.train_file)
		self.vertices_train = self.vertices_train[:-self.nVal]
		self.vertices_val = self.vertices_train[-self.nVal:]

		self.n_vertex = self.vertices_train.shape[1]

		self.vertices_test = np.load(self.test_file)
		self.vertices_test = self.vertices_test

	def normalize(self):
		self.vertices_train = self.vertices_train - self.mean
		self.vertices_train = self.vertices_train/self.std

		self.vertices_val = self.vertices_val - self.mean
		self.vertices_val = self.vertices_val/self.std

		self.vertices_test = self.vertices_test - self.mean
		self.vertices_test = self.vertices_test/self.std

		self.N = self.vertices_train.shape[0]

		# self.pca.fit(self.vertices_train)
		# eigenVals = np.sqrt(self.pca.explained_variance_)
		# self.pcaMatrix = np.dot(np.diag(eigenVals), self.pca.components_)
		print('Vertices normalized')

	def vec2mesh(self, vec):
		vec = vec.reshape((self.n_vertex, 3))*self.std + self.mean
		return Mesh(v=vec, f=self.reference_mesh.f)

	def show(self, ids):
		'''ids: list of ids to play '''
		if max(ids)>=self.N:
			raise ValueError('id: out of bounds')

		mesh = Mesh(v=self.vertices_train[ids[0]], f=self.reference_mesh.f)
		time.sleep(0.5)    # pause 0.5 seconds
		viewer = mesh.show()
		for i in range(len(ids)-1):
			viewer.dynamic_meshes = [Mesh(v=self.vertices_train[ids[i+1]], f=self.reference_mesh.f)]
			time.sleep(0.5)    # pause 0.5 seconds
		return 0

	def sample(self, BATCH_SIZE=64):
		datasamples = np.zeros((BATCH_SIZE, self.vertices_train.shape[1]*self.vertices_train.shape[2]))
		for i in range(BATCH_SIZE):
			_randid = random.randint(0,self.N-1)
			#print _randid
			datasamples[i] = ((deepcopy(self.vertices_train[_randid]) - self.mean)/self.std).reshape(-1)

		return datasamples

	def save_meshes(self, filename, meshes):
		for i in range(meshes.shape[0]):
			vertices = meshes[i].reshape((self.n_vertex, 3))*self.std + self.mean
			mesh = Mesh(v=vertices, f=self.reference_mesh.f)
			mesh.write_ply(filename+'-'+str(i).zfill(3)+'.ply')
		return 0

	def show_mesh(self, viewer, mesh_vec1, mesh_vec2):
		mesh1 = Mesh(v=mesh_vec1.reshape((self.n_vertex,3))*self.std + self.mean, f=self.reference_mesh.f)
		mesh2 = Mesh(v=mesh_vec2.reshape((self.n_vertex,3))*self.std + self.mean, f=self.reference_mesh.f)

		viewer[0][0].set_dynamic_meshes([mesh1])
		viewer[0][1].set_dynamic_meshes([mesh2])
		time.sleep(0.1)    # pause 0.5 seconds
		return 0

def meshPlay(folder,every=100,wait=0.05):
	files = glob.glob(folder+'/*')
	files.sort()
	files = files[-1000:]
	view = MeshViewer()
	for i in range(0,len(files),every):
		mesh = Mesh(filename=files[i])
		view.dynamic_meshes = [mesh]
		time.sleep(wait)

class MakeSlicedTimeDataset(object):
	"""docstring for FaceMesh"""
	def __init__(self, folders, dataset_name):
		self.facial_motion_dirs = folders
		self.dataset_name = dataset_name

		self.train_datapaths = self.gather_paths("train")
		self.test_datapaths = self.gather_paths("test")

		self.train_vertices = self.gather_data(self.train_datapaths)
		self.test_vertices = self.gather_data(self.test_datapaths)
		self.save_vertices()

	def gather_paths(self, opt):
		datapaths = []
		for i in range(len(self.facial_motion_dirs)):
			print(self.facial_motion_dirs[i])
			datapaths += glob.glob(self.facial_motion_dirs[i]+'/*/*/*.ply')

		trainpaths = []
		testpaths = []
		for i in range(len(datapaths)):
			if (i%100) < 10:
				testpaths += [datapaths[i]]
				#print(datapaths[i])
			else:
				trainpaths += [datapaths[i]]

		if opt=="train":
			print opt+" data of size: ", len(trainpaths)
			#print(trainpaths[:10])
			return trainpaths
		if opt=="test":
			print opt+" data of size: ", len(testpaths)
			return testpaths

	def gather_data(self, datapaths):
		vertices = []
		for p in tqdm(datapaths):
			mesh_file = p
			#print "Loading ", mesh_file
			face_mesh = Mesh(filename=mesh_file)
			#print "Loaded ", mesh_file
			vertices.append(face_mesh.v)
		return np.array(vertices)

	def save_vertices(self):
		if not os.path.exists(self.dataset_name):
			os.makedirs(self.dataset_name)
		np.save(self.dataset_name+'/train', self.train_vertices)
		np.save(self.dataset_name+'/test', self.test_vertices)

		print "Saving ... ", self.dataset_name+'/train'
		print "Saving ... ", self.dataset_name+'/test'
		return 0

class MakeIdentityExpressionDataset(object):
	"""docstring for FaceMesh"""
	def __init__(self, folders, test_exp, dataset_name,crossval="expression", use_templates=0):
		self.facial_motion_dirs = folders
		self.dataset_name = dataset_name
		self.test_exp = test_exp
		self.crossval = crossval


		self.train_datapaths = self.gather_paths("train")
		self.test_datapaths = self.gather_paths("test")

		self.train_vertices = self.gather_data(self.train_datapaths)
		self.test_vertices = self.gather_data(self.test_datapaths)
		self.save_vertices()

	def gather_paths(self, opt):
		datapaths = []
		for i in range(len(self.facial_motion_dirs)):
			print(self.facial_motion_dirs[i])
			datapaths += glob.glob(self.facial_motion_dirs[i]+'/*/*/*.ply')

		trainpaths = []
		testpaths = []

		if self.crossval=="expression":
			path_flagger = -2
		if self.crossval == "identity":
			path_flagger = -3

		for i in range(len(datapaths)):
			p = datapaths[i]
			if p.split('/')[path_flagger] == self.test_exp:
				testpaths += [datapaths[i]]
				#print(datapaths[i])
			else:
				trainpaths += [datapaths[i]]

		if opt=="train":
			print opt+" data of size: ", len(trainpaths)
			#print(trainpaths[:10])
			return trainpaths
		if opt=="test":
			print opt+" data of size: ", len(testpaths)
			return testpaths

	def gather_data(self, datapaths):
		vertices = []
		labels = []
		for p in tqdm(datapaths):
			mesh_file = p
			#print "Loading ", mesh_file
			face_mesh = Mesh(filename=mesh_file)
			face_mesh_v = face_mesh.v
			vertices.append(face_mesh_v)
		return np.array(vertices)

	def save_vertices(self):
		if not os.path.exists(self.dataset_name):
			os.makedirs(self.dataset_name)
		np.save(self.dataset_name+'/train', self.train_vertices)
		np.save(self.dataset_name+'/test', self.test_vertices)

		print "Saving ... ", self.dataset_name+'/train'
		print "Saving ... ", self.dataset_name+'/test'
		return 0

def generateSlicedTimeDataSet(data_path, save_path):
	MakeSlicedTimeDataset(folders=[data_path], dataset_name=os.path.join(save_path, 'sliced'))
	return 0

def generateExpressionDataSet(data_path, save_path):
	test_exps = ['bareteeth','cheeks_in','eyebrow','high_smile','lips_back','lips_up','mouth_down',
				'mouth_extreme','mouth_middle','mouth_open','mouth_side','mouth_up']

	for exp in test_exps:
		fm = MakeIdentityExpressionDataset(folders=[data_path], test_exp=exp, dataset_name=os.path.join(save_path, exp), use_templates=0)

def generateIdentityDataset(data_path, save_path):
	test_ids = ['FaceTalk_170725_00137_TA',  'FaceTalk_170731_00024_TA',  'FaceTalk_170811_03274_TA',
	  			'FaceTalk_170904_00128_TA',  'FaceTalk_170908_03277_TA',  'FaceTalk_170913_03279_TA',
				'FaceTalk_170728_03272_TA',  'FaceTalk_170809_00138_TA',  'FaceTalk_170811_03275_TA',
				'FaceTalk_170904_03276_TA',  'FaceTalk_170912_03278_TA',  'FaceTalk_170915_00223_TA']

	for ids in test_ids:
		fm = MakeIdentityExpressionDataset(folders=[data_path], test_exp=ids, dataset_name=os.path.join(save_path,ids), crossval="identity", use_templates=0)
