using System;
using System.Collections.Generic;
using System.Text;
using FaceRecognitionDotNet.Extensions.SkLearn.Metrics;
using FaceRecognitionDotNet.Extensions.SkLearn.Utils;

// ReSharper disable once CheckNamespace
namespace FaceRecognitionDotNet.SkLearn.Neighbors
{

    public class ClassifierMixin
    {

        // """Mixin class for all classifiers in scikit-learn."""
        // _estimator_type = "classifier"
        // 
        public void score(X, y, sample_weight= None)
        {
            // """Returns the mean accuracy on the given test data and labels.
            // 
            // In multi-label classification, this is the subset accuracy
            // which is a harsh metric since you require for each sample that
            // each label set be correctly predicted.
            // 
            // Parameters
            // ----------
            // X : array-like, shape = (n_samples, n_features)
            // Test samples.
            // 
            // y : array-like, shape = (n_samples)or(n_samples, n_outputs)
            // True labels for X.
            // 
            // sample_weight : array-like, shape = [n_samples], optional
            // Sample weights.
            // 
            // Returns
            // -------
            // score : float
            // Mean accuracy of self.predict(X) wrt. y.
            // 
            // """
            return Classification.accuracy_score(y, self.predict(X), sample_weight = sample_weight)
        }

    }

}