from facemesh import FaceData
import numpy as np
import argparse
import time
import readchar
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
def cumsum_error(error, n_bins=1000):
	N_cnn, X_cnn = np.histogram(error, bins=n_bins )
	X_cnn = np.convolve(X_cnn, 0.5*np.ones(2))
	X_cnn = X_cnn[1:-1]

	factor_cnn = 100.0/error.shape[0]
	cumsum_N_cnn = np.cumsum(N_cnn)

	X_vec_cnn = np.zeros((n_bins+1,))
	X_vec_cnn[1:] = X_cnn

	yVec_cnn = np.zeros((n_bins+1,))
	yVec_cnn[1:] = factor_cnn*cumsum_N_cnn

	return X_vec_cnn, yVec_cnn

def save_error_plt(cnn, pca, test,fname,n_bins=1000):
	cnn_err = np.sqrt(np.sum((cnn - test)**2, axis=2))
	pca_err = np.sqrt(np.sum((pca - test)**2, axis=2))
	cnn_err = np.reshape(cnn_err, (-1,))
	pca_err = np.reshape(pca_err, (-1,))

	X_vec_pca, yVec_pca = cumsum_error(pca_err)
	X_vec_cnn, yVec_cnn = cumsum_error(cnn_err)

	plt.plot(X_vec_pca, yVec_pca, label="PCA")
	plt.plot(X_vec_cnn, yVec_cnn, label="Mesh Autoencoder")
	plt.ylabel('Percentage of Vertices')
	plt.xlabel('Euclidean Error norm (mm)')

	plt.legend(loc='lower right')
	plt.xlim(0,6)
	plt.grid(True)#,color='grey', linestyle='-', linewidth=0.5)
	plt.savefig(fname)



parser = argparse.ArgumentParser()
parser.add_argument('--cnn', default='/is/ps/shared/aranjan/convFaces/results/iclr/bareteeth21_predictions.npy', help='path to dataset')
parser.add_argument('--data', default='/is/ps/shared/aranjan/convFaces/data/bareteeth', help='path to dataset')
parser.add_argument('--nz', type=int, default=8, help='size of the latent z vector')
parser.add_argument('--save', default='save', help='path to dataset')

opt = parser.parse_args()

if not os.path.exists(opt.save):
    os.makedirs(opt.save)
    os.makedirs(opt.save+'/cnn')
    os.makedirs(opt.save+'/pca')
    os.makedirs(opt.save+'/gt')
    os.makedirs(opt.save+'/plt')

reference_mesh_file = 'data/template.obj'
facedata = FaceData(nVal=100, train_file=opt.data+'/train.npy',
    test_file=opt.data+'/test.npy', reference_mesh_file=reference_mesh_file, pca_n_comp=opt.nz, fitpca=True)
nv = facedata.n_vertex*3
nTest = facedata.vertices_test.shape[0]

cnn_outputs = np.load(opt.cnn)
pca_outputs = facedata.pca.inverse_transform(facedata.pca.transform(np.reshape(facedata.vertices_test, (facedata.vertices_test.shape[0], facedata.n_vertex*3))))

cnn_vertices = (cnn_outputs * facedata.std) + facedata.mean
pca_vertices = (np.reshape(pca_outputs,(pca_outputs.shape[0],facedata.n_vertex, 3)) * facedata.std) + facedata.mean
test_vertices = (facedata.vertices_test * facedata.std) + facedata.mean

cnn_vertices = cnn_vertices*1000
pca_vertices = pca_vertices*1000
test_vertices = test_vertices*1000

save_error_plt(cnn=cnn_vertices,pca=pca_vertices, test=test_vertices, fname=opt.save+'/plt/'+os.path.basename(opt.data)+'.png')
cnn_err = np.mean(np.sqrt(np.sum((cnn_vertices-test_vertices)**2, axis=2)))
pca_err = np.mean(np.sqrt(np.sum((pca_vertices-test_vertices)**2, axis=2)))

cnn_std = np.std(np.sqrt(np.sum((cnn_vertices-test_vertices)**2, axis=2)))
pca_std = np.std(np.sqrt(np.sum((pca_vertices-test_vertices)**2, axis=2)))

cnn_med = np.median(np.sqrt(np.sum((cnn_vertices-test_vertices)**2, axis=2)))
pca_med = np.median(np.sqrt(np.sum((pca_vertices-test_vertices)**2, axis=2)))

print('{} PCA Error: {:.3f}+{:.3f} | {:.3f} CoMA Error: {:.3f}+{:.3f} | {:.3f}'.format(os.path.basename(opt.data), pca_err, pca_std, pca_med, cnn_err, cnn_std,  cnn_med))

