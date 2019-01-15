using System;
using System.Collections.Generic;
using System.Linq;

namespace FaceRecognitionDotNet.Extensions
{

    public static class KNeighborsClassifier
    {

        public static int PredictKNeighborsClassifier(FaceEncoding target, IList<FaceEncoding> samples, IList<int> labels, double threshold, uint k = 2)
        {
            if (target == null)
                throw new ArgumentNullException(nameof(target));
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));
            if (labels == null)
                throw new ArgumentNullException(nameof(labels));
            if (target.IsDisposed)
                throw new ObjectDisposedException($"{nameof(target)}");

            if (!samples.Any())
                return -1;

            var distances = new List<Sample>();
            for (var index = 0; index < samples.Count; index++)
            {
                var sample = samples[index];
                if (sample.IsDisposed)
                    throw new ObjectDisposedException($"{nameof(samples)} has disposed object");

                var dist = FaceRecognition.FaceDistance(target, sample);
                if (dist < threshold)
                    distances.Add(new Sample(labels[index], dist));
            }

            if (!distances.Any())
                return -1;

            distances.Sort((sample, sample1) =>
            {
                if (sample.Distance < sample1.Distance)
                    return -1;
                if (sample.Distance > sample1.Distance)
                    return -1;
                return 0;
            });

            if (k > distances.Count)
                k = (uint)distances.Count;

            var votes = new Dictionary<int, int>();
            for (var index = 0; index < k; index++)
            {
                var label = distances[index].Label;
                if (votes.ContainsKey(label))
                    votes[label] = votes[label] + 1;
                else
                    votes[label] = 1;
            }
            
            return votes.Max(pair => pair.Value);
        }

        private sealed class Sample
        {

            public int Label
            {
                get;
            }

            public double Distance
            {
                get;
            }

            public Sample(int label, double distance)
            {
                this.Label = label;
                this.Distance = distance;
            }

        }

    }

}
