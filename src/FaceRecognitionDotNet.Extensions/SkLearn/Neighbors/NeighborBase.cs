using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{

    internal class NeighborBase
    {

        protected int n_neighbors = 0;

        protected double radius;

        protected int leaf_size;

        protected Dictionary<string, string> metric_params;

        protected int p;

        protected int n_jobs;

        protected string metric;

        protected string algorithm;

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

        private void _fit(IList<double[]> X)
        {
            this._check_algorithm_metric();
            if (this.metric_params == null)
                this.effective_metric_params_ = { }:
            else
                this.effective_metric_params_ = this.metric_params.copy();

            var effective_p = this.effective_metric_params_.get('p', this.p);
            if (this.metric in ["wminkowski", "minkowski"])
                this.effective_metric_params_['p'] = effective_p;

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
                    this.effective_metric_params_['p'] = p;
            }

            if (isinstance(X, NeighborsBase))
            {
                this._fit_X = X._fit_X;
                this._tree = X._tree;
                this._fit_method = X._fit_method;
                return this;
            }
            else if (isinstance(X, BallTree))
            {
                this._fit_X = X.data;
                this._tree = X;
                this._fit_method = "ball_tree";
                return this;
            }
            else if (isinstance(X, KDTree))
            {
                this._fit_X = X.data;
                this._tree = X;
                this._fit_method = "kd_tree";
                return this;
            }

            X = check_array(X, accept_sparse = "csr");

            var n_samples = X.shape[0]
            if (n_samples == 0)
                throw new Exception("n_samples must be greater than 0");

            if (issparse(X))
            {
                if (this.algorithm not in ("auto", "brute"))
                    Console.WriteLine("cannot use tree with sparse input: using brute force");
                if (this.effective_metric_ not in VALID_METRICS_SPARSE["brute"] and not callable(this.effective_metric_))
                    throw new Exception($"Metric '{this.effective_metric}' not valid for sparse input. " +
                                         "Use sorted(sklearn.neighbors." +
                                         "VALID_METRICS_SPARSE["brute"]) " +
                                         "to get valid options. " +
                                         "Metric can also be a callable function.");
                    this._fit_X = X.copy();
                this._tree = None;
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
                    if (this.effective_metric_ in VALID_METRICS["kd_tree"])
                        this._fit_method = "kd_tree";
                    else if ((callable(this.effective_metric_) or this.effective_metric_ in VALID_METRICS["ball_tree"]);
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
                    this._tree = BallTree(X, this.leaf_size,
                        metric = this.effective_metric_,
                        **this.effective_metric_params_);
                    break;
                case "kd_tree":
                    this._tree = KDTree(X, this.leaf_size,
                        metric = this.effective_metric_,
                        **this.effective_metric_params_);
                    break;
                case "brute":
                    this._tree = None;
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
