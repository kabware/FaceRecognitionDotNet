using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{
    public class KNeighborsMixin
    {
        // Mixin for k-neighbors searches

        public void _kneighbors_reduce_func(dist, start, int n_neighbors, return_distance)
        {
            // Reduce a chunk of distances to the nearest neighbors
            // 
            // Callback to :func:`sklearn.metrics.pairwise.pairwise_distances_chunked`
            // 
            // Parameters
            // ----------
            // dist : array of shape (n_samples_chunk, n_samples)
            // start : int
            //     The index in X which the first row of dist corresponds to.
            // n_neighbors : int
            // return_distance : bool
            // 
            // Returns
            // -------
            // dist : array of shape (n_samples_chunk, n_neighbors), optional
            //     Returned only if return_distance
            // neigh : array of shape (n_samples_chunk, n_neighbors)
            // """
            sample_range = np.arange(dist.shape[0])[:, None];
            var neigh_ind = np.argpartition(dist, n_neighbors - 1, axis = 1);
            neigh_ind = neigh_ind[:, :n_neighbors];
            // argpartition doesn't guarantee sorted order, so we sort again
            neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])];
            if (return_distance)
                if (this.effective_metric_ == "euclidean")
                    result = np.sqrt(dist[sample_range, neigh_ind]), neigh_ind;
                else
                    result = dist[sample_range, neigh_ind], neigh_ind;
            else
                result = neigh_ind;

            return result;
        }

        public void kneighbors(X = None, n_neighbors= None, return_distance= True)
        {
            // Finds the K-neighbors of a point.
            // Returns indices of and distances to the neighbors of each point.
            // 
            // Parameters
            // ----------
            // X : array-like, shape (n_query, n_features), \
            //         or (n_query, n_indexed) if metric == 'precomputed'
            //     The query point or points.
            //     If not provided, neighbors of each indexed point are returned.
            //     In this case, the query point is not considered its own neighbor.
            // 
            // n_neighbors : int
            //     Number of neighbors to get (default is the value
            //     passed to the constructor).
            // 
            // return_distance : boolean, optional. Defaults to True.
            //     If False, distances will not be returned
            // 
            // Returns
            // -------
            // dist : array
            //     Array representing the lengths to points, only present if
            //     return_distance=True
            // 
            // ind : array
            //     Indices of the nearest points in the population matrix.
            // 
            // Examples
            // --------
            // In the following example, we construct a NeighborsClassifier
            // class from an array representing our data set and ask who's
            // the closest point to [1,1,1]
            // 
            // >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
            // >>> from sklearn.neighbors import NearestNeighbors
            // >>> neigh = NearestNeighbors(n_neighbors=1)
            // >>> neigh.fit(samples) // doctest: +ELLIPSIS
            // NearestNeighbors(algorithm='auto', leaf_size=30, ...)
            // >>> print(neigh.kneighbors([[1., 1., 1.]])) // doctest: +ELLIPSIS
            // (array([[0.5]]), array([[2]]))
            // 
            // As you can see, it returns [[0.5]], and [[2]], which means that the
            // element is at distance 0.5 and is the third element of samples
            // (indexes start at 0). You can also query for multiple points:
            // 
            // >>> X = [[0., 1., 0.], [1., 0., 1.]]
            // >>> neigh.kneighbors(X, return_distance=False) // doctest: +ELLIPSIS
            // array([[1],
            //        [2]]...)
            // 
            // """
            check_is_fitted("_fit_method");


            if (n_neighbors == null)
                n_neighbors = this.n_neighbors;
            else if (n_neighbors <= 0)
                throw new Exception($"Expected n_neighbors > 0. Got {n_neighbors}");
            else
                if (!np.issubdtype(type(n_neighbors), np.integer))
                throw new Exception($"n_neighbors does not take {type(n_neighbors)} value, enter integer value");

            if (X != null)
            {
                query_is_train = false;
                X = check_array(X, accept_sparse = "csr");
            }
            else
            {
                query_is_train = true;
                X = this._fit_X;
                // Include an extra neighbor to account for the sample itself being
                // returned, which is removed later
                n_neighbors += 1;
            }


            train_size = this._fit_X.shape[0];
            if (n_neighbors > train_size)
                throw new Exception($"Expected n_neighbors <= n_samples, but n_samples = {train_size}, n_neighbors = {n_neighbors}");

            n_samples, _ = X.shape;
            sample_range = np.arange(n_samples)[:, None];


            n_jobs = effective_n_jobs(this.n_jobs);
            if (this._fit_method == "brute")
            {
                reduce_func = partial(this._kneighbors_reduce_func,
                    n_neighbors = n_neighbors,
                    return_distance = return_distance)

                // for efficiency, use squared euclidean distances
                kwds = ({ 'squared': True}
                if (this.effective_metric_ == "euclidean") else this.effective_metric_params_)

                result = list(pairwise_distances_chunked(
                    X, this._fit_X, reduce_func = reduce_func,
                    metric = this.effective_metric_, n_jobs = n_jobs,
                    **kwds));
            }

            else if (new[] { "ball_tree", "kd_tree" }.Contains(this._fit_method))
            {
                if (issparse(X))
                    throw new Exception($"{this._fit_method} does not work with sparse matrices. Densify the data, or set algorithm='brute'");
                old_joblib = LooseVersion(joblib_version) < LooseVersion('0.12');
                if (sys.version_info < (3, ) || old_joblib)
                {
                    // Deal with change of API in joblib
                    check_pickle = old_joblib ? false : null;
                    delayed_query = delayed(_tree_query_parallel_helper, check_pickle = check_pickle);
                    parallel_kwargs =  { "backend": "threading"};
                }
                else
                {
                    delayed_query = delayed(_tree_query_parallel_helper);
                    parallel_kwargs =  { "prefer": "threads"};
                }

                result = Parallel(n_jobs, **parallel_kwargs)(
                    delayed_query(this._tree, X[s], n_neighbors, return_distance)for s in
                    gen_even_slices(X.shape[0], n_jobs)
                ) ;
            }
            else
                throw new Exception("internal: _fit_method not recognized");

            if (return_distance)
            {
                dist, neigh_ind = zip(*result);
                result = np.vstack(dist), np.vstack(neigh_ind);
            }
            else
            {
                result = np.vstack(result);
            }

            if (!query_is_train)
                return result;
            else
            {
                // If the query data is the same as the indexed data, we would like
                // to ignore the first nearest neighbor of every sample, i.e
                // the sample itself.
                if (return_distance)
                    dist, neigh_ind = result;
                else
                    neigh_ind = result;

                sample_mask = neigh_ind != sample_range;

                // Corner case: When the number of duplicates are more
                // than the number of neighbors, the first NN will not
                // be the sample, but a duplicate.
                // In that case mask the first duplicate.
                dup_gr_nbrs = np.all(sample_mask, axis = 1);
                sample_mask[:, 0][dup_gr_nbrs] = false;

                neigh_ind = np.reshape(neigh_ind[sample_mask], (n_samples, n_neighbors - 1));

                if (return_distance)
                {
                    dist = np.reshape(dist[sample_mask], (n_samples, n_neighbors - 1));
                    return dist, neigh_ind;
                }

                return neigh_ind;
            }
        }

        public void kneighbors_graph(X= None, n_neighbors= None, mode= "connectivity")
        {
            // Computes the (weighted) graph of k-Neighbors for points in X

            // Parameters
            // ----------
            // X : array-like, shape (n_query, n_features), \
            //         or (n_query, n_indexed) if metric == 'precomputed'
            //     The query point or points.
            //     If not provided, neighbors of each indexed point are returned.
            //     In this case, the query point is not considered its own neighbor.
            // 
            // n_neighbors : int
            //     Number of neighbors for each sample.
            //     (default is value passed to the constructor).
            // 
            // mode : {'connectivity', 'distance'}, optional
            //     Type of returned matrix: 'connectivity' will return the
            //     connectivity matrix with ones and zeros, in 'distance' the
            //     edges are Euclidean distance between points.
            // 
            // Returns
            // -------
            // A : sparse matrix in CSR format, shape = [n_samples, n_samples_fit]
            //     n_samples_fit is the number of samples in the fitted data
            //     A[i, j] is assigned the weight of edge that connects i to j.
            // 
            // Examples
            // --------
            // >>> X = [[0], [3], [1]]
            // >>> from sklearn.neighbors import NearestNeighbors
            // >>> neigh = NearestNeighbors(n_neighbors=2)
            // >>> neigh.fit(X) // doctest: +ELLIPSIS
            // NearestNeighbors(algorithm='auto', leaf_size=30, ...)
            // >>> A = neigh.kneighbors_graph(X)
            // >>> A.toarray()
            // array([[1., 0., 1.],
            //        [0., 1., 1.],
            //        [1., 0., 1.]])
            // 
            // See also
            // --------
            // NearestNeighbors.radius_neighbors_graph
            // """
            if (n_neighbors != null)
                n_neighbors = this.n_neighbors;
    
            // kneighbors does the None handling.
            if (X != null)
            {
                X = check_array(X, accept_sparse = "csr");
                n_samples1 = X.shape[0];
            }
            else
            {
                n_samples1 = this._fit_X.shape[0];
            }
            
            n_samples2 = this._fit_X.shape[0];
            n_nonzero = n_samples1 * n_neighbors;
            A_indptr = np.arange(0, n_nonzero + 1, n_neighbors);
    
            // construct CSR matrix representation of the k-NN graph
            if (mode == "connectivity")
            {
                A_data = np.ones(n_samples1 * n_neighbors);
                A_ind = this.kneighbors(X, n_neighbors, return_distance = False);
            }
            else if (mode == "distance")
            {
                A_data, A_ind = this.kneighbors(X, n_neighbors, return_distance = True);
                A_data = np.ravel(A_data);
            }
            else
                throw new Exception($"Unsupported mode, must be one of 'connectivity'" +
                                     "or 'distance' but got '{mode}' instead");

            kneighbors_graph = csr_matrix((A_data, A_ind.ravel(), A_indptr), shape = (n_samples1, n_samples2));

            return kneighbors_graph;
        }

    }
}
