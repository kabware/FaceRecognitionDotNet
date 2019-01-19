namespace FaceRecognitionDotNet.Extensions.SkLearn.Neighbors
{

    internal sealed class BallTree
    {

        public static string[] valid_metrics =
        {
            "EuclideanDistance", "SEuclideanDistance",
            "ManhattanDistance", "ChebyshevDistance",
            "MinkowskiDistance", "WMinkowskiDistance",
            "MahalanobisDistance", "HammingDistance",
            "CanberraDistance", "BrayCurtisDistance",
            "JaccardDistance", "MatchingDistance",
            "DiceDistance", "KulsinskiDistance",
            "RogersTanimotoDistance", "RussellRaoDistance",
            "SokalMichenerDistance", "SokalSneathDistance",
            "PyFuncDistance", "HaversineDistance"
        };

    }

}