using System;
using System.Collections.Generic;
using System.Text;

namespace FaceRecognitionDotNet.Extensions.SkLearn.Metrics
{

    internal sealed class Pairwise
    {

        public static readonly IDictionary<string, string[]> PAIRWISE_DISTANCE_FUNCTIONS;

        static Pairwise()
        {
            PAIRWISE_DISTANCE_FUNCTIONS = new Dictionary<string, string[]>();
            PAIRWISE_DISTANCE_FUNCTIONS.Add("cityblock", new[] { "manhattan_distances" });
            PAIRWISE_DISTANCE_FUNCTIONS.Add("cosine", new[] { "cosine_distances" });
            PAIRWISE_DISTANCE_FUNCTIONS.Add("euclidean", new[] { "euclidean_distances" });
            PAIRWISE_DISTANCE_FUNCTIONS.Add("l2", new[] { "euclidean_distances" });
            PAIRWISE_DISTANCE_FUNCTIONS.Add("l1", new[] { "manhattan_distances" });
            PAIRWISE_DISTANCE_FUNCTIONS.Add("manhattan", new[] { "manhattan_distances" });
            PAIRWISE_DISTANCE_FUNCTIONS.Add("precomputed", new[] { "None" });
        }

    }

}
