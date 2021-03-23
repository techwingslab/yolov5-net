using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using Yolov5Net.Scorer;
using Yolov5Net.Scorer.Models;

namespace Yolov5Net.App
{
    class Program
    {
        static void Main(string[] args)
        {
            var scorer = new YoloScorer<YoloCocoModel>();

            using var stream = new FileStream("assets/test.jpg", FileMode.Open);

            var image = Image.FromStream(stream);

            List<YoloPrediction> predictions = scorer.Predict(image);

            using var graphics = Graphics.FromImage(image);

            foreach (var prediction in predictions) // iterate each prediction to draw results
            {
                double score = Math.Round(prediction.Score, 2);

                graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
                    new[] { prediction.Rectangle });

                var (x, y) = (prediction.Rectangle.X - 2, prediction.Rectangle.Y - 21);

                graphics.DrawString($"{prediction.Label.Name} ({score})", new Font("Consolas", 3),
                    new SolidBrush(prediction.Label.Color), new PointF(x, y));
            }

            image.Save("assets/result.jpg");
        }
    }
}
