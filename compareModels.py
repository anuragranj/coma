from facemesh import FaceData
from psbody.mesh import MeshViewers
import numpy as np
import argparse
import time
import readchar

parser = argparse.ArgumentParser()
parser.add_argument('--cnn', default='.', help='path to dataset')
parser.add_argument('--data', default='allset', help='path to dataset')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')

opt = parser.parse_args()

reference_mesh_file = 'data/template.obj'
facedata = FaceData(nVal=1, train_file=opt.data+'_train.npy',
    test_file=opt.data+'_test.npy', reference_mesh_file=reference_mesh_file, pca_n_comp=int(opt.nz))
nv = facedata.n_vertex*3
nTest = facedata.vertices_test.shape[0]


cnn_outputs = np.load(opt.cnn)
pca_outputs = facedata.pca.inverse_transform(facedata.pca.transform(facedata.vertices_test))

cnn_vertices = (cnn_outputs * facedata.std) + facedata.mean
pca_vertices = (np.reshape(pca_outputs,(pca_outputs.shape[0],facedata.n_vertex, 3)) * facedata.std) + facedata.mean
test_vertices = (np.reshape(facedata.vertices_test, (facedata.vertices_test.shape[0], facedata.n_vertex, 3)) * facedata.std) + facedata.mean

viewer = MeshViewers(window_width=800, window_height=200, shape=[1, 4], titlebar='Meshes')

i = np.random.randint(0,nTest-1)
while(1):

	outmesh = np.zeros((4,nv))
	outmesh[0] = np.reshape(cnn_vertices[i], (nv,))
	outmesh[1] = np.reshape(pca_vertices[i], (nv,))
	outmesh[3] = np.reshape(test_vertices[i], (nv,))

	facedata.show_mesh(viewer=viewer, mesh_vecs=outmesh, figsize=(1,4), normalize=False)
	inp = readchar.readchar()
	if inp=="+":
		i = i+1
	else:
		i = i-1
	i = i%(nTest-1)
	print(i)
