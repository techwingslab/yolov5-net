using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;

using var image = await Image.LoadAsync<Rgba32>("Assets/test.jpg");

using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5n.onnx");

var predictions = scorer.Predict(image);

var font = new Font(new FontCollection().Add(@"C:\Windows\Fonts\consola.ttf"), 16);
foreach (var prediction in predictions) // iterate predictions to draw results
{
    var score = Math.Round(prediction.Score, 2);

    var (x, y) = (prediction.Rectangle.Left - 3, prediction.Rectangle.Top - 23);

    //var (w, h) = (image.Width, image.Height);
    var (w, h) = (640, 640);
    image.Mutate(a => a.DrawPolygon(new Pen(prediction.Label.Color, 1),
        new PointF(prediction.Rectangle.Left / 640 * w, prediction.Rectangle.Top / 640 * h),
        new PointF(prediction.Rectangle.Right, prediction.Rectangle.Top),
        new PointF(prediction.Rectangle.Right, prediction.Rectangle.Bottom),
        new PointF(prediction.Rectangle.Left, prediction.Rectangle.Bottom)
    ).DrawText($"{prediction.Label.Name} ({score})",
        font,
        prediction.Label.Color,
        new(x, y)));
}

await image.SaveAsync("Assets/result.jpg");
