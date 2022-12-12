using SixLabors.ImageSharp;

namespace Yolov5Net.Scorer;

/// <summary>
/// Object prediction.
/// </summary>
public record YoloPrediction(YoloLabel Label, float Score, RectangleF Rectangle);
