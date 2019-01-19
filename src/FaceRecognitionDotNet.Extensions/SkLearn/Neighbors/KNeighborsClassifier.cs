using System;
using System.Collections.Generic;
using System.Text;
using FaceRecognitionDotNet.Extensions.SkLearn.Utils;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{

    internal sealed class KNeighborsClassifier : NeighborsBase
    // (NeighborsBase, KNeighborsMixin,
    // SupervisedIntegerMixin, ClassifierMixin)
    {

        #region Fields

        private string weights;

        #endregion

        #region Properties
        #endregion

        #region Methods

        #region Overrids
        #endregion

        #region Event Handlers
        #endregion

        #region Helpers
        #endregion

        #endregion

        //    """Classifier implementing the k-nearest neighbors vote.

        //    Read more in the :ref:`User Guide<classification>`.

        //    Parameters
        //    ----------
        //    n_neighbors : int, optional (default = 5)
        //        Number of neighbors to use by default for :meth:`kneighbors` queries.

        //    weights : str or callable, optional(default = 'uniform')
        //        weight function used in prediction.Possible values:

        //        - 'uniform' : uniform weights.All points in each neighborhood
        //          are weighted equally.
        //        - 'distance' : weight points by the inverse of their distance.
        //          in this case, closer neighbors of a query point will have a
        //          greater influence than neighbors which are further away.
        //        - [callable] : a user-defined function which accepts an
        //          array of distances, and returns an array of the same shape
        //          containing the weights.

        //    algorithm : { 'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        //Algorithm used to compute the nearest neighbors:

        //        - 'ball_tree' will use :class:`BallTree`
        //        - 'kd_tree' will use :class:`KDTree`
        //        - 'brute' will use a brute-force search.
        //        - 'auto' will attempt to decide the most appropriate algorithm
        //          based on the values passed to :meth:`fit` method.

        //        Note: fitting on sparse input will override the setting of
        //        this parameter, using brute force.

        //    leaf_size : int, optional(default = 30)
        //        Leaf size passed to BallTree or KDTree.This can affect the
        //        speed of the construction and query, as well as the memory
        //        required to store the tree.The optimal value depends on the
        //        nature of the problem.

        //    p : integer, optional (default = 2)
        //        Power parameter for the Minkowski metric.When p = 1, this is
        //        equivalent to using manhattan_distance (l1), and euclidean_distance
        //        (l2) for p = 2. For arbitrary p, minkowski_distance(l_p) is used.

        //   metric : string or callable, default 'minkowski'
        //        the distance metric to use for the tree.The default metric is
        //        minkowski, and with p= 2 is equivalent to the standard Euclidean
        //        metric. See the documentation of the DistanceMetric class for a
        //        list of available metrics.

        //    metric_params : dict, optional(default = None)
        //        Additional keyword arguments for the metric function.

        //    n_jobs : int or None, optional(default=None)
        //        The number of parallel jobs to run for neighbors search.
        //        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        //        ``-1`` means using all processors.See :term:`Glossary<n_jobs>`
        //        for more details.
        //        Doesn't affect :meth:`fit` method.

        //    Examples
        //    --------
        //    >>> X = [[0], [1], [2], [3]]
        //    >>> y = [0, 0, 1, 1]
        //    >>> from sklearn.neighbors import KNeighborsClassifier
        //    >>> neigh = KNeighborsClassifier(n_neighbors = 3)
        //    >>> neigh.fit(X, y) # doctest: +ELLIPSIS
        //    KNeighborsClassifier(...)
        //    >>> print(neigh.predict([[1.1]]))
        //    [0]
        //    >>> print(neigh.predict_proba([[0.9]]))
        //    [[0.66666667 0.33333333]]

        //    See also
        //    --------
        //    RadiusNeighborsClassifier
        //    KNeighborsRegressor
        //    RadiusNeighborsRegressor
        //    NearestNeighbors

        //    Notes
        //    -----
        //    See :ref:`Nearest Neighbors<neighbors>` in the online documentation
        //    for a discussion of the choice of ``algorithm`` and ``leaf_size``.

        //    .. warning::

        //       Regarding the Nearest Neighbors algorithms, if it is found that two
        //       neighbors, neighbor `k+1` and `k`, have identical distances
        //       but different labels, the results will depend on the ordering of the
        //       training data.

        //    https://en.wikipedia.org/wiki/K-nearest_neighbor_algorithm
        //    """
        #region Constructors

        public KNeighborsClassifier(int n_neighbors = 5,
            string weights = "uniform",
            string algorithm = "auto",
            int leaf_size = 30,
            int p = 2,
            string metric = "minkowski",
            object metric_params = null,
            object n_jobs = null) :
            base(n_neighbors, algorithm, leaf_size, p, metric, metric_params, n_jobs)
        {
            this.weights = _check_weights(weights);
        }

        #endregion

        public void predict(IEnumerable<double[]> X)
        {
            // """Predict the class labels for the provided data

            //Parameters
            //----------
            //X: array - like, shape(n_query, n_features), \
            //or(n_query, n_indexed) if metric == 'precomputed'
            //Test samples.

            //            Returns
            //        ------ -
            //    y : array of shape[n_samples] or[n_samples, n_outputs]
            //Class labels for each data sample.
            //"""
            X = Validation.check_array(X, accept_sparse = "csr");

            var neigh_dist, neigh_ind = this.kneighbors(X);
            var classes_ = this.classes_;
            var _y = this._y;
            if (!this.outputs_2d_)
            {
                _y = this._y.reshape((-1, 1));
                classes_ = [this.classes_];
            }

            var n_outputs = len(classes_);
            var n_samples = X.shape[0];
            weights = _get_weights(neigh_dist, this.weights);

            y_pred = np.empty((n_samples, n_outputs), dtype = classes_[0].dtype);
            for k, classes_k in enumerate(classes_)
            {
                if (weights == null)
                    mode, _ = stats.mode(_y[neigh_ind, k], axis = 1);
                else
                    mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis = 1);

                mode = np.asarray(mode.ravel(), dtype = np.intp);
                y_pred[:, k] = classes_k.take(mode);
            }

            if (!this.outputs_2d_)
                y_pred = y_pred.ravel();

            return y_pred;
        }

        public void predict_proba(IEnumerable<double[]> X)
        {

            // """Return probability estimates for the test data X.

            //Parameters
            //----------
            //X: array - like, shape(n_query, n_features), \
            //or(n_query, n_indexed) if metric == 'precomputed'
            //Test samples.

            //            Returns
            //        ------ -
            //    p : array of shape = [n_samples, n_classes], or a list of n_outputs
            //    of such arrays if n_outputs > 1.
            //    The class probabilities of the input samples.Classes are ordered
            //    by lexicographic order.
            //"""
            X = Validation.check_array(X, accept_sparse = "csr");

            var neigh_dist, neigh_ind = this.kneighbors(X);

            var classes_ = this.classes_;
            var _y = this._y;
            if (!this.outputs_2d_)
            {
                _y = this._y.reshape((-1, 1));
                classes_ = [this.classes_];
            }

            var n_samples = X.shape[0];

            weights = _get_weights(neigh_dist, this.weights);
            if (weights == null)
                weights = np.ones_like(neigh_ind);

            all_rows = np.arange(X.shape[0]);
            probabilities = [];
            for k, classes_k in enumerate(classes_)
            {
                pred_labels = _y[:, k][neigh_ind];
                proba_k = np.zeros((n_samples, classes_k.size));

                // # a simple ':' index doesn't work right
                for i, idx in enumerate(pred_labels.T)  // loop is O(n_neighbors)
                    proba_k[all_rows, idx] += weights[:, i];

                // # normalize 'votes' into real [0,1] probabilities
                normalizer = proba_k.sum(axis = 1)[:, np.newaxis];
                normalizer[normalizer == 0.0] = 1.0;
                proba_k /= normalizer;

                probabilities.append(proba_k);
            }

            if (!this.outputs_2d_)
                probabilities = probabilities[0];

            return probabilities;
        }

    }

}