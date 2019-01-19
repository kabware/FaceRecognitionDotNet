using System.Collections.Generic;
using System.Linq;
using FaceRecognitionDotNet.Extensions.SkLearn.Metrics;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{

    internal sealed class Base
    {

        public static readonly IDictionary<string, string[]> VALID_METRICS;

        public static readonly IDictionary<string, string[]> VALID_METRICS_SPARSE;

        static Base()
        {
            VALID_METRICS = new Dictionary<string, string[]>();
            VALID_METRICS.Add("ball_tree", BallTree.valid_metrics);
            VALID_METRICS.Add("kd_tree", KDTree.valid_metrics);
            var keys = Pairwise.PAIRWISE_DISTANCE_FUNCTIONS.Keys;
            var value = new[] { "braycurtis",
                "canberra",
                "chebyshev",
                "correlation",
                "cosine",
                "dice",
                "hamming",
                "jaccard",
                "kulsinski",
                "mahalanobis",
                "matching",
                "minkowski",
                "rogerstanimoto",
                "russellrao",
                "seuclidean",
                "sokalmichener",
                "sokalsneath",
                "sqeuclidean",
                "yule",
                "wminkowski" };
            var tmp = new List<string>();
            tmp.AddRange(keys);
            tmp.AddRange(value);
            VALID_METRICS.Add("brute", tmp.ToArray());

            VALID_METRICS_SPARSE = new Dictionary<string, string[]>();
            VALID_METRICS_SPARSE.Add("ball_tree", new string[0]);
            VALID_METRICS_SPARSE.Add("kd_tree", new string[0]);
            VALID_METRICS_SPARSE.Add("brute", Pairwise.PAIRWISE_DISTANCE_FUNCTIONS.Keys.ToArray());
        }
    }

}