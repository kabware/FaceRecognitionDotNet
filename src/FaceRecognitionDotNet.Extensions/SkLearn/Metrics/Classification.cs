using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Metrics
{

    public class Classification
    {

        public static void accuracy_score(y_true, y_pred, bool normalize= true, string sample_weight= null)
        {
            // """Accuracy classification score.
            // 
            // In multilabel classification, this function computes subset accuracy:
            // the set of labels predicted for a sample must* exactly* match the
            // corresponding set of labels in y_true.
            // 
            // Read more in the :ref:`User Guide<accuracy_score>`.
            // 
            // Parameters
            // ----------
            // y_true : 1d array-like, or label indicator array / sparse matrix
            //     Ground truth (correct) labels.
            // 
            // y_pred : 1d array-like, or label indicator array / sparse matrix
            //     Predicted labels, as returned by a classifier.
            // 
            // normalize : bool, optional (default=True)
            //     If ``False``, return the number of correctly classified samples.
            //     Otherwise, return the fraction of correctly classified samples.
            // 
            // sample_weight : array-like of shape = [n_samples], optional
            //     Sample weights.
            // 
            // Returns
            // -------
            // score : float
            //     If ``normalize == True``, return the fraction of correctly
            //     classified samples (float), else returns the number of correctly
            //     classified samples (int).
            // 
            //     The best performance is 1 with ``normalize == True`` and the number
            //     of samples with ``normalize == False``.
            // 
            // See also
            // --------
            // jaccard_similarity_score, hamming_loss, zero_one_loss
            // 
            // Notes
            // -----
            // In binary and multiclass classification, this function is equal
            // to the ``jaccard_similarity_score`` function.
            // 
            // Examples
            // --------
            // >>> import numpy as np
            // >>> from sklearn.metrics import accuracy_score
            // >>> y_pred = [0, 2, 1, 3]
            // >>> y_true = [0, 1, 2, 3]
            // >>> accuracy_score(y_true, y_pred)
            // 0.5
            // >>> accuracy_score(y_true, y_pred, normalize= False)
            // 2
            // 
            // In the multilabel case with binary label indicators:
            // 
            // >>> accuracy_score(np.array([[0, 1], [1, 1]]), np.ones((2, 2)))
            // 0.5
            // """
            // 
            // Compute accuracy for each possible representation
            y_type, y_true, y_pred = Classification._check_targets(y_true, y_pred);
            check_consistent_length(y_true, y_pred, sample_weight);
            if (y_type.startswith("multilabel"))
            {
                differing_labels = count_nonzero(y_true - y_pred, axis = 1);
                score = differing_labels == 0;
            }
            else
            {
                score = y_true == y_pred;
            }

            return _weighted_sum(score, sample_weight, normalize);
        }

        private static void _check_targets(y_true, y_pred)
        {

            // """Check that y_true and y_pred belong to the same classification task
            // 
            // This converts multiclass or binary types to a common shape, and raises a
            // ValueError for a mix of multilabel and multiclass targets, a mix of
            // multilabel formats, for the presence of continuous-valued or multioutput
            // targets, or for targets of different lengths.
            // 
            // Column vectors are squeezed to 1d, while multilabel formats are returned
            // as CSR sparse label indicators.
            // 
            // Parameters
            // ----------
            // y_true : array-like
            // 
            // y_pred : array-like
            // 
            // Returns
            // -------
            // type_true : one of { 'multilabel-indicator', 'multiclass', 'binary'}
            //     The type of the true target data, as output by
            //     ``utils.multiclass.type_of_target``
            // 
            // y_true : array or indicator matrix
            // 
            // y_pred : array or indicator matrix
            // """
            check_consistent_length(y_true, y_pred)
            type_true = type_of_target(y_true)
            type_pred = type_of_target(y_pred)

            y_type = set([type_true, type_pred])
            if y_type == set(["binary", "multiclass"]) :
                y_type = set(["multiclass"])

            if len(y_type) > 1:
                raise ValueError("Classification metrics can't handle a mix of {0} "
                                 "and {1} targets".format(type_true, type_pred))

            # We can't have more than one value on y_type => The set is no more needed
                y_type = y_type.pop()

            # No metrics support "multiclass-multioutput" format
                    if (y_type not in ["binary", "multiclass", "multilabel-indicator"]):
                raise ValueError("{0} is not supported".format(y_type))

            if y_type in ["binary", "multiclass"]:
                y_true = column_or_1d(y_true)
                y_pred = column_or_1d(y_pred)
                if y_type == "binary":
                    unique_values = np.union1d(y_true, y_pred)
                    if len(unique_values) > 2:
                        y_type = "multiclass"

            if y_type.startswith('multilabel'):
                y_true = csr_matrix(y_true)
                y_pred = csr_matrix(y_pred)
                y_type = 'multilabel-indicator'

            return y_type, y_true, y_pred
        }

        private double _weighted_sum(double[] sample_score, string sample_weight, bool normalize= false)
        {
            if (normalize)
                return np.average(sample_score, weights = sample_weight);
            else if (sample_weight != null)
                return np.dot(sample_score, sample_weight);
            else
                return sample_score.Sum();
        }

    }

}