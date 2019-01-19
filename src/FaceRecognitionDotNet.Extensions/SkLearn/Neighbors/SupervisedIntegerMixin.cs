using System;
using System.Collections.Generic;
using System.Text;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{

    public class SupervisedIntegerMixin
    {

        protected bool outputs_2d_;

        private string _fit_method;

        private string metric;

        private string effective_metric_;

        public void fit<T>(IList<double[]> X, T[] y)
        {
            // """Fit the model using X as training data and y as target values
            // 
            // Parameters
            // ----------
            // X : {array-like, sparse matrix, BallTree, KDTree}
            // Training data.If array or matrix, shape[n_samples, n_features],
            //     or[n_samples, n_samples] if metric= 'precomputed'.
            // 
            // y : { array - like, sparse matrix}
            // Target values of shape = [n_samples] or[n_samples, n_outputs]
            // 
            // """
            if not isinstance(X, (KDTree, BallTree)):
                X, y = check_X_y(X, y, "csr", multi_output = True)

            if (y.Rank == 1 || y.Rank == 2 && y.shape[1] == 1)
            {
                if (y.Rank != 1)
                {
                    Console.WriteLine("A column-vector y was passed when a 1d array " +
                                      "was expected. Please change the shape of y to " +
                                      "(n_samples, ), for example using ravel().");
                }

                this.outputs_2d_ = false;
                y = y.reshape((-1, 1));
            }
            else
            {
                this.outputs_2d_ = true;
            }

            check_classification_targets(y)
            this.classes_ = []
            this._y = np.empty(y.shape, dtype = np.int)
            for k in range(this._y.shape[1])
            {
                classes, this._y[:, k] = np.unique(y[:, k], return_inverse = True)
                this.classes_.append(classes)
            }

            if (!this.outputs_2d_)
            {
                this.classes_ = this.classes_[0];
                this._y = this._y.ravel();
            }

            return this._fit(X);
        }
        
    }

}
