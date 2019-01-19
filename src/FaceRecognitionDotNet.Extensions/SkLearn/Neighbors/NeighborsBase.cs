using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using FaceRecognitionDotNet.Extensions.SkLearn.Utils;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{

    internal class NeighborsBase
    {

        #region Fields

        protected int n_neighbors = 0;

        protected double radius;

        protected int leaf_size;

        protected Dictionary<string, string> metric_params;

        protected int p;

        protected int n_jobs;

        protected string metric;

        protected string algorithm;

        #endregion

        #region Constructors

        protected NeighborsBase(int n_neighbors = 5,
                                string algorithm = "auto",
                                int leaf_size = 30,
                                int p = 2,
                                string metric = "minkowski",
                                object metric_params = null,
                                object n_jobs = null)
        {
            this.n_neighbors = n_neighbors;
            this.algorithm = algorithm;
            this.p = p;
            this.leaf_size = leaf_size;
            this.metric = metric;
            this.metric_params = this.metric_params;
            this.n_jobs = this.n_jobs;
        }

        #endregion

        #region Properties
        #endregion

        #region Methods

        #region Overrids
        #endregion

        #region Event Handlers
        #endregion

        #region Helpers
        
        protected string _check_weights(string weights)
        {
            if (new[] { null, "uniform", "distance" }.Contains(weights))
                return weights;
            if callable(weights)
                return weights;
            
            throw new Exception("weights not recognized: should be 'uniform', 'distance', or a callable function");
        }

        protected string _get_weights(object dist, string weights)
        {
            if (new[] { null, "uniform"}.Contains(weights))
                return null;

            if (weights == "distance")
            {
                // if user attempts to classify a point that was zero distance from one
                // or more training points, those training points are weighted as 1.0
                // and the other points as 0.0
                if (dist.dtype is np.dtype(object))
                {
                    foreach(point_dist_i, point_dist in enumerate(dist))
                    {
                        // check if point_dist is iterable
                        // (ex: RadiusNeighborClassifier.predict may set an element of
                        // dist to 1e-6 to represent an 'outlier')
                        if (hasattr(point_dist, "__contains__") && 0. in point_dist)
                            dist[point_dist_i] = point_dist == 0.;
                        else
                            dist[point_dist_i] = 1. / point_dist;
                    }
                }
                else
                {
                    with np.errstate(divide = "ignore"):
                        dist = 1. / dist;
                    inf_mask = np.isinf(dist);
                    inf_row = np.any(inf_mask, axis = 1);
                    dist[inf_row] = inf_mask[inf_row];
                }

                return dist
            }

            if (callable(weights))
                return weights(dist);

            throw new Exception("weights not recognized: should be 'uniform', 'distance', or a callable function");
        }

        #endregion

        #endregion

        protected void _check_algorithm_metric()
        {
            if (!new[] { "auto", "brute", "kd_tree", "ball_tree" }.Contains(this.algorithm))
                throw new Exception($"unrecognized algorithm: '{this.algorithm}'");

            string alg_check;
            if (this.algorithm == "auto")
            {
                if (this.metric == "precomputed")
                    alg_check = "brute";
                else if (callable(this.metric) || Base.VALID_METRICS["ball_tree"].Contains(this.metric))
                    alg_check = "ball_tree";
                else
                    alg_check = "brute";
            }
            else
            {
                alg_check = this.algorithm;
            }

            if (callable(this.metric))
            {
                if (this.algorithm == "kd_tree")
                    // callable metric is only valid for brute force and ball_tree
                    throw new Exception($"kd_tree algorithm does not support callable metric '{this.metric}'");
            }
            else if (!Base.VALID_METRICS[alg_check].Contains(this.metric))
            {
                throw new Exception($"Metric '{this.metric}' not valid. Use " +
                                    "sorted(sklearn.neighbors.VALID_METRICS['{alg_check}']) " +
                                    "to get valid options. " +
                                    "Metric can also be a callable function.");
            }

            int effective_p;
            if (this.metric_params != null && this.metric_params.ContainsKey("p"))
            {
                Console.WriteLine("Parameter p is found in metric_params. " +
                                  "The corresponding parameter from __init__ " +
                                  "is ignored.");
                effective_p = int.Parse(this.metric_params["p"]);
            }
            else
            {
                effective_p = this.p;
            }

            if (new[] { "wminkowski", "minkowski" }.Contains(this.metric) && effective_p < 1)
                throw new Exception("p must be greater than one for minkowski metric");
        }

        private Dictionary<string, int> effective_metric_params_;

        private string effective_metric_;

        private string _fit_method;

        private object _tree;

        private void _fit(IList<double[]> X)
        {
            this._check_algorithm_metric();
            if (this.metric_params == null)
                this.effective_metric_params_ = new Dictionary<string, int>();
            else
                this.effective_metric_params_ = this.metric_params.copy();

            var effective_p = this.effective_metric_params_.get('p', this.p);
            if (new []{ "wminkowski", "minkowski" }.Contains(this.metric))
                this.effective_metric_params_["p"] = effective_p;

            this.effective_metric_ = this.metric;
            // For minkowski distance, use more efficient methods where available
            if (this.metric == "minkowski")
            {
                var p = this.effective_metric_params_.pop('p', 2);
                if (p < 1)
                    throw new Exception("p must be greater than one for minkowski metric");
                else if (p == 1)
                    this.effective_metric_ = "manhattan";
                else if (p == 2)
                    this.effective_metric_ = "euclidean";
                else if (p == np.inf)
                    this.effective_metric_ = "chebyshev";
                else
                    this.effective_metric_params_["p"] = p;
            }
            
            if (X is NeighborsBase)
            {
                var tmp = (NeighborsBase)X;
                this._fit_X = tmp._fit_X;
                this._tree = tmp._tree;
                this._fit_method = tmp._fit_method;
                return this;
            }
            else if (X is BallTree)
            {
                var tmp = (BallTree)X;
                this._fit_X = tmp.data;
                this._tree = tmp;
                this._fit_method = "ball_tree";
                return this;
            }
            else if (X is KDTree)
            {
                var tmp = (KDTree)X;
                this._fit_X = tmp.data;
                this._tree = tmp;
                this._fit_method = "kd_tree";
                return this;
            }
            
            X = Validation.check_array(X, accept_sparse = "csr");

            var n_samples = X.shape[0];
            if (n_samples == 0)
                throw new Exception("n_samples must be greater than 0");

            if (issparse(X))
            {
                if (!new []{ "auto", "brute" }.Contains(this.algorithm))
                    Console.WriteLine("cannot use tree with sparse input: using brute force");
                if (!Base.VALID_METRICS_SPARSE["brute"].Contains(this.effective_metric_) && not callable(this.effective_metric_))
                    throw new Exception($"Metric '{this.effective_metric}' not valid for sparse input. " +
                                         "Use sorted(sklearn.neighbors." +
                                         "VALID_METRICS_SPARSE['brute']) " +
                                         "to get valid options. " +
                                         "Metric can also be a callable function.");
                this._fit_X = X.copy();
                this._tree = null;
                this._fit_method = "brute";
                return this;
            }

            this._fit_method = this.algorithm;
            this._fit_X = X;

            if (this._fit_method == "auto")
            {
                // A tree approach is better for small number of neighbors,
                // and KDTree is generally faster when available
                if ((this.n_neighbors == null || this.n_neighbors < this._fit_X.shape[0] / 2) && this.metric != "precomputed")
                {
                    if (Base.VALID_METRICS["kd_tree"].Contains(this.effective_metric_))
                        this._fit_method = "kd_tree";
                    else if (callable(this.effective_metric_) || Base.VALID_METRICS["ball_tree"].Contains(this.effective_metric_))
                        this._fit_method = "ball_tree";
                    else
                        this._fit_method = "brute";
                }
                else
                {
                    this._fit_method = "brute";
                }
            }

            switch (this._fit_method)
            {
                case "ball_tree":
                    this._tree = new BallTree(X,
                                              this.leaf_size,
                                              metric = this.effective_metric_,
                                              **this.effective_metric_params_);
                    break;
                case "kd_tree":
                    this._tree = new KDTree(X, 
                                            this.leaf_size,
                                            metric = this.effective_metric_,
                                            **this.effective_metric_params_);
                    break;
                case "brute":
                    this._tree = null;
                    break;
                default:
                    throw new Exception($"algorithm = '{this.algorithm}' not recognized");
            }

            if (this.n_neighbors != null)
                if (this.n_neighbors <= 0)
                    throw new Exception($"Expected n_neighbors > 0. Got {this.n_neighbors}");
                else
                    if (not np.issubdtype(type(this.n_neighbors), np.integer))
                        throw new Exception($"n_neighbors does not take {type(this.n_neighbors)} value, enter integer value");

            return this;
        }

    }

}