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
            var image = Image.FromFile("Assets/test.jpg");

            byte[] weights = File.ReadAllBytes("Assets/Weights/yolov5s6.onnx");

            var scorer = new YoloScorer<YoloCocoP6Model>();

            List<YoloPrediction> predictions = scorer.Predict(image, weights);

            using (var graphics = Graphics.FromImage(image))
            {
                foreach (var prediction in predictions) // iterate predictions to draw results
                {
                    double score = Math.Round(prediction.Score, 2);

                    graphics.DrawRectangles(new Pen(prediction.Label.Color, 1),
                        new[] { prediction.Rectangle });

                    var (x, y) = (prediction.Rectangle.X - 3, prediction.Rectangle.Y - 23);

                    graphics.DrawString($"{prediction.Label.Name} ({score})",
                        new Font("Consolas", 16, GraphicsUnit.Pixel), new SolidBrush(prediction.Label.Color),
                        new PointF(x, y));
                }

                image.Save("Assets/result.jpg");
            }
        }
    }
}
