# This module contains functions to create data sets from simulations
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from multiprocessing import Pool
from sklearn.neighbors import radius_neighbors_graph
from .utils import sparse_to_tuple, rand_rotation_matrix
 

def project_ellipticities(i, data):

    a3d = np.array([[data['dm_av_x'][i], data['dm_av_y'][i], data['dm_av_z'][i]]])
    b3d = np.array([[data['dm_bv_x'][i], data['dm_bv_y'][i], data['dm_bv_z'][i]]])
    c3d = np.array([[data['dm_cv_x'][i], data['dm_cv_y'][i], data['dm_cv_z'][i]]])
    q3d = np.array([data['dm_b'][i]/data['dm_a'][i]])
    s3d = np.array([data['dm_c'][i]/data['dm_a'][i]])


    e12 = project_3d_shape(a3d, b3d, c3d, q3d, s3d)

    return e12

def project_ellipticities_np(i, data):

    a3d = np.array([[data['dm_av_x'][i], data['dm_av_y'][i], data['dm_av_z'][i]]])
    b3d = np.array([[data['dm_bv_x'][i], data['dm_bv_y'][i], data['dm_bv_z'][i]]])
    c3d = np.array([[data['dm_cv_x'][i], data['dm_cv_y'][i], data['dm_cv_z'][i]]])
    q3d = np.array([data['dm_b'][i]/data['dm_a'][i]])
    s3d = np.array([data['dm_c'][i]/data['dm_a'][i]])


    e12 = project_3d_shape_np(a3d, b3d, c3d, q3d, s3d)

    return e12

# def project_3d_shape(a3d, b3d, c3d, q3d, s3d):
 
 
#     s = tf.stack([a3d, b3d, c3d])
#     w = tf.stack([tf.ones_like(q3d), q3d, s3d])
 

#     k = tf.reduce_sum(s[:,:,0:2]*tf.expand_dims(s[:,:,2], axis=-1) / tf.expand_dims(w[:,:]**2, axis=-1), axis=0)
#     a2 =tf.reduce_sum(s[:,:,2]**2/w[:,:]**2, axis=0)
 
#     Winv = tf.reduce_sum(tf.einsum('ijk,ijl->ijkl', s[:,:,0:2], s[:,:,0:2]) / tf.expand_dims(tf.expand_dims(w[:,:]**2,-1),-1), axis=0) - tf.einsum('ij,ik->ijk', k,k)/tf.expand_dims(tf.expand_dims(a2,-1),-1)
    
#     W = tf.linalg.inv(Winv )
#     d = tf.sqrt(tf.linalg.det(W))
 
#     e1 = (W[:,0,0] - W[:,1,1])/( W[:,0,0] + W[:,1,1] + 2*d)
#     e2 = 2 * W[:,0,1]/( W[:,0,0] + W[:,1,1] + 2*d)
 
#     return tf.stack([e1, e2], axis=-1)

def project_3d_shape(a3d, b3d, c3d, q3d, s3d):
 
 
    s = tf.stack([a3d, b3d, c3d])
    #print('Look it: ',s.get_shape().as_list())
    w = tf.stack([tf.ones_like(q3d), q3d, s3d])
 

    k = tf.reduce_sum(s[:,:,0:2]*tf.expand_dims(s[:,:,2], axis=-2) / tf.expand_dims(w[:,:]**2, axis=-2), axis=0)
    a2 =tf.reduce_sum(s[:,:,2]**2/w[:,:]**2, axis=0)
    #print('Look it: ',s[:,:,0:2,...].get_shape().as_list())
 
    
    Winv = tf.reduce_sum(tf.einsum('ijko,ijlo->ijklo', s[:,:,0:2,...], s[:,:,0:2,...]) / tf.expand_dims(tf.expand_dims(w[:,:]**2,-2),-2), axis=0) - tf.einsum('ijo,iko->ijko', k,k)/tf.expand_dims(tf.expand_dims(a2,-2),-2)

    
    
    W = tf.linalg.pinv(tf.squeeze( tf.transpose(Winv) ))
    

#     W = tf.where(tf.math.is_nan(W), tf.ones_like(W), W)
#     W = tf.where(tf.math.is_inf(W), tf.ones_like(W), W) 
    
    d = tf.sqrt( tf.math.abs(tf.linalg.det(W) ))
    
#     d = tf.where(tf.math.is_nan(d), tf.ones_like(d), d)
#     d = tf.where(tf.math.is_inf(d), tf.ones_like(d), d)
 
    denom = ( W[:,0,0] + W[:,1,1] + 2.0*d)
    e1 = (W[:,0,0] - W[:,1,1])/denom 
    e2 = 2.0 * W[:,0,1]/denom 
#         denom = tf.where(tf.math.is_nan(denom), tf.ones_like(denom), denom)
#         denom = tf.where(tf.math.is_inf(denom), tf.ones_like(denom), denom)
#         denom = tf.where( denom < 0.00001 , 1.0, denom)
    
    #e1 = tf.math.divide_no_nan( (W[:,0,0] - W[:,1,1]),denom )
    #e2 = tf.math.divide_no_nan( 2.0 * W[:,0,1],denom )
   # e1 = tf.ones_like(q3d)
  #  e2 = tf.ones_like(q3d)
 #   e1 = tf.where(e1==0, 1.0, e1 )
#    e2 = tf.where(e2==0, 1.0, e2)
    #print('Got to return')
    
#     #check for infs and nans
    e1 = tf.where(tf.math.is_nan(e1), tf.zeros_like(e1), e1)
    e2 = tf.where(tf.math.is_nan(e2), tf.zeros_like(e2), e2)
    
    e1 = tf.where(tf.math.is_inf(e1), tf.zeros_like(e1), e1)
    e2 = tf.where(tf.math.is_inf(e2), tf.zeros_like(e2), e2)
    #make zero length vec
    e1 = tf.where(tf.math.is_nan(e2), tf.zeros_like(e2), e1)
    e2 = tf.where(tf.math.is_nan(e1), tf.zeros_like(e1), e2)
   
    
    #make zero length vec
    e1 = tf.where(tf.math.is_inf(e2), tf.zeros_like(e2), e1)
    e2 = tf.where(tf.math.is_inf(e1), tf.zeros_like(e1), e2)
    
    return tf.stack([e1, e2], axis=-1)

def project_3d_shape_np(a3d, b3d, c3d, q3d, s3d):
 
 
    s = np.stack([a3d, b3d, c3d])
    w = np.stack([np.ones_like(q3d), q3d, s3d])
 

    k = np.sum(s[:,:,0:2]*np.expand_dims(s[:,:,2], axis=-1) / np.expand_dims(w[:,:]**2, axis=-1), axis=0)
    a2 =np.sum(s[:,:,2]**2/w[:,:]**2, axis=0)
     
    Winv = np.sum(np.einsum('ijk,ijl->ijkl', s[:,:,0:2], s[:,:,0:2]) / np.expand_dims(np.expand_dims(w[:,:]**2,-1),-1), axis=0) - np.einsum('ij,ik->ijk', k,k)/np.expand_dims(np.expand_dims(a2,-1),-1)
 
    W = np.linalg.inv(Winv )
    d = np.sqrt(np.linalg.det(W))
 
    e1 = (W[:,0,0] - W[:,1,1])/( W[:,0,0] + W[:,1,1] + 2*d)
    e2 = 2 * W[:,0,1]/( W[:,0,0] + W[:,1,1] + 2*d)
 
    return np.stack([e1, e2], axis=-1)


def _process_graph(args):
    """
    Function that preprocesses the dataset for fast training
    """
    gid, group_ids, Xsp, X, Y, n_features, n_labels, graph_radius = args
    g = np.where(group_ids == gid)[0]
    xsp = Xsp[g]
    x = X[g]
    y = Y[g]
    # Compute adjacency matrix for each entry
    graph = radius_neighbors_graph(xsp, graph_radius, mode='connectivity',
                                   include_self=False)

    return (xsp, x, y, graph)

def graph_input_fn(catalog,
                   vector_features=(), scalar_features=(),
                   vector_labels=(), scalar_labels=(),
                   pos_key=['gal_pos_x', 'gal_pos_y', 'gal_pos_z'],
                   group_key='GroupID', batch_size=128,
                   noise_size=32,
                   graph_radius=1000., shuffle=False, repeat=False,
                   prefetch=100, poolsize=12, balance_key='group_mass_scaled',
                   rotate=False):
    """
    Python generator function that will create batches of graphs from
    input catalog.
    """
    
    features = vector_features + scalar_features
    labels = vector_labels + scalar_labels

    # It takes a minute but we precompute all the graphs and data
    # Identify the individual groups and pre-extract the relevant data
    group_ids = catalog[group_key]
    gids, idx = np.unique(group_ids, return_index=True)
 
    # Extracts columns of interest into memory first
    Xsp = np.array(catalog[pos_key]).view(np.float64).reshape((-1, 3)).astype(np.float32)
    X = np.array(catalog[features]).view(np.float64).reshape((-1, len(features))).astype(np.float32)
    Y = np.array(catalog[labels]).view(np.float64).reshape((-1, len(labels))).astype(np.float32)
    #nan_mask = (~np.isnan(Xsp).any(axis=1)) & (~np.isnan(X).any(axis=1)) & (~np.isnan(Y).any(axis=1))
    Xsp = Xsp[ (~np.isnan(Xsp).any(axis=1))]
    X = X[(~np.isnan(X).any(axis=1))]
    Y = Y[(~np.isnan(Y).any(axis=1))]
 
    n_batches = len(gids) // batch_size
    last_batch = len(gids) % batch_size

    n_features = len(vector_features) // 3
    n_labels = len(vector_labels) // 3

    print("Precomputing dataset")
    with Pool(poolsize) as p:
        cache = p.map(_process_graph, [(gid, group_ids, Xsp, X, Y, n_features, n_labels, graph_radius) for gid in gids])
    print("Done")

    if balance_key is not None:
        # Balance probablities of graphs based on group mass
        p, bin = np.histogram(catalog[balance_key][idx], 16)
        mbin = np.digitize(catalog[balance_key][idx], bin[:-1]) - 1
        cat_probs = (1./p)[mbin]
        cat_probs /= cat_probs.sum()

    def graph_generator():

        while True:
            # Apply permutation
            #shuffling matters for training and we want to use shuffled data for training.
            #not for the testing 
            
            if shuffle:
                if balance_key is not None:
                    batch_gids = np.random.choice(len(gids), len(gids), p=cat_probs)
                else:
                    batch_gids = np.random.permutation(len(gids))
            else:
                batch_gids = range(len(gids))

            for b in range(n_batches+1):
                if b == n_batches:
                    bs = last_batch
                else:
                    bs = batch_size

                # Extract the groupId for the elements of the batch
                inds = batch_gids[batch_size*b:batch_size*b + bs]

                res = [cache[i] for i in inds]

                graphs = [r[-1] for r in res]
                xsp = np.concatenate([r[0] for r in res])
                x = np.concatenate([r[1] for r in res])
                y = np.concatenate([r[2] for r in res])
                n = np.random.randn(len(x), noise_size).astype(np.float32)

                # Apply rotation of vector quantities if requested
                if rotate:
                    M = rand_rotation_matrix()
                    xsp = xsp.dot(M.T)
                    for i in range(n_features):
                       # print(i)
                        x[:, i*3:i*3+3] = x[:, i*3:i*3+3].dot(M.T)
                       # print(x[:, i*3:i*3+3])
                    for i in range(n_labels):
                        #print(i)
                        y[:, i*3:i*3+3] = y[:, i*3:i*3+3].dot(M.T)
                        #print(y[:, i*3:i*3+3])
                    #for j in each_gal:
                        #e1,e2= somefunction_2d(y[j])
#                 print(x.shape)
                    #print(y.shape)
#                 print(x)
#                 print(y)
                # Block adjacency matrix for the batch
                W = sp.block_diag(graphs)

                # Building pooling matrix for the batch
                data = np.concatenate([np.ones(graphs[i].shape[0])/graphs[i].shape[0] for i in range(len(graphs))])
                row = np.concatenate([a*np.ones(graphs[i].shape[0]) for a,i in enumerate(range(len(graphs)))]).astype('int')
                col = np.arange(W.shape[0]).astype('int32')

                # Preparing sparse matrices in TF format
                pooling_matrix = sparse_to_tuple(sp.coo_matrix((data, (row,col)), shape=(bs, W.shape[0])))
                W = sparse_to_tuple(W)

                yield (W[0], W[1], W[2],
                       pooling_matrix[0], pooling_matrix[1], pooling_matrix[2],
                       xsp, x, n), y

            if not repeat:
                break

    graph_generator.output_types = ((tf.int32, tf.float32, tf.int64,
                                     tf.int32, tf.float32, tf.int64,
                                     tf.float32, tf.float32, tf.float32),
                                    tf.float32)
     
    graph_generator.output_shapes = (((None, 2), (None,), (2,),
                                      (None, 2), (None,), (2,),
                                      (None, 3), (None, len(features)),
                                      (None, noise_size)),
                                     (None, len(labels)))
     
    dataset = tf.data.Dataset.from_generator(graph_generator,
                               output_types = graph_generator.output_types,
                               output_shapes = graph_generator.output_shapes)
    dataset = dataset.prefetch(64)
    return dataset

def sim_input_fn(catalog,
                   vector_features=(), scalar_features=(),
                   vector_labels=(), scalar_labels=(),
                   batch_size=128, shuffle=False, repeat=False,
                   poolsize=12, balance_key='mass',
                   rotate=False):
    """
    Python generator function that will create batches of graphs from
    input catalog.
    """
    features = vector_features + scalar_features
    labels = vector_labels + scalar_labels

    # It takes a minute but we precompute all the graphs and data
    # Identify the individual groups and pre-extract the relevant data
    X = np.array(catalog[features]).view(np.float64).reshape((-1, len(features))).astype(np.float32)
    Y = np.array(catalog[labels]).view(np.float64).reshape((-1, len(labels))).astype(np.float32)

    n_batches = len(catalog) // batch_size
    last_batch = len(catalog) % batch_size

    n_features = len(vector_features) // 3
    n_labels = len(vector_labels) // 3

    if balance_key is not None:
        # Balance probablities of graphs based on group mass
        idx = np.arange(len(catalog))
        p, bin = np.histogram(catalog[balance_key][idx], 16)
        mbin = np.digitize(catalog[balance_key][idx], bin[:-1]) - 1
        cat_probs = (1./p)[mbin]
        cat_probs /= cat_probs.sum()

    def batch_generator():

        while True:
            # Apply permutation
            if shuffle:
                if balance_key is not None:
                    batch_gids = np.random.choice(len(catalog), len(catalog), p=cat_probs)
                else:
                    batch_gids = np.random.permutation(len(catalog))
            else:
                batch_gids = range(len(catalog))

            for b in range(n_batches+1):
                if b == n_batches:
                    bs = last_batch
                else:
                    bs = batch_size

                # Extract the elements of the batch
                inds = batch_gids[batch_size*b:batch_size*b + bs]
                x = X[inds]
                y = Y[inds]

                # Apply rotation of vector quantities if requested
                if rotate:
                    M = rand_rotation_matrix()
                    for i in range(n_features):
                        x[:, i*3:i*3+3] = x[:, i*3:i*3+3].dot(M.T)
                    for i in range(n_labels):
                        y[:, i*3:i*3+3] = y[:, i*3:i*3+3].dot(M.T)

                yield x, y # {k: x[:,i] for i,k in enumerate(features)}, {k: y[:,i] for i,k in enumerate(labels)}

            if not repeat:
                break

    batch_generator.output_types = (tf.float32, tf.float32)
    batch_generator.output_shapes = ((None, len(features)), (None, len(labels)))

    dataset = tf.data.Dataset.from_generator(batch_generator,
                               output_types = batch_generator.output_types,
                               output_shapes = batch_generator.output_shapes)
    dataset = dataset.prefetch(64)
    return dataset
