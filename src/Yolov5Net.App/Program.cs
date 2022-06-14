using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Threading.Tasks;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;

namespace Yolov5Net.App
{
    internal class Program
    {
        private static async Task Main(string[] args)
        {
            using var image = await Image.LoadAsync<Rgba32>("Assets/test.jpg");

            using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5n.onnx");

            var predictions = scorer.Predict(image);

            var font = new Font(new FontCollection().Add(@"C:\Windows\Fonts\consola.ttf"), 16);
            foreach (var prediction in predictions) // iterate predictions to draw results
            {
                var score = Math.Round(prediction.Score, 2);

                var (x, y) = (prediction.Rectangle.Left - 3, prediction.Rectangle.Top - 23);

                image.Mutate(a => a.DrawPolygon(new Pen(prediction.Label.Color, (float)score),
                    new PointF(prediction.Rectangle.Left, prediction.Rectangle.Top),
                    new PointF(prediction.Rectangle.Right, prediction.Rectangle.Top),
                    new PointF(prediction.Rectangle.Right, prediction.Rectangle.Bottom),
                    new PointF(prediction.Rectangle.Left, prediction.Rectangle.Bottom)
                ).DrawText($"{prediction.Label.Name} ({score})",
                    font,
                    prediction.Label.Color,
                    new PointF(x, y)));
            }

            await image.SaveAsync("Assets/result.jpg");
        }

        //static void Main(string[] args)
        //{
        //    using var image = Image.FromFile("Assets/test.jpg");

        //    using var scorer = new YoloScorer<YoloCocoP5Model>("Assets/Weights/yolov5n.onnx");

        //    List<YoloPrediction> predictions = scorer.Predict(image);

        //    using var graphics = Graphics.FromImage(image);

        //    foreach (var prediction in predictions) // iterate predictions to draw results
        //    {
        //        double score = Math.Round(prediction.Score, 2);

        //        graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
        //            new[] { prediction.Rectangle });

        //        var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

        //        graphics.DrawString($"{prediction.Label.Name} ({score})",
        //            new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
        //            new PointF(x, y));
        //    }

        //    image.Save("Assets/result.jpg");
        //}
    }
}
