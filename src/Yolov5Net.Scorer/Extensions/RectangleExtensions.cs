using SixLabors.ImageSharp;

namespace Yolov5Net.Scorer.Extensions;

public static class RectangleExtensions
{
    public static float Area(this RectangleF source) => source.Width * source.Height;
}
