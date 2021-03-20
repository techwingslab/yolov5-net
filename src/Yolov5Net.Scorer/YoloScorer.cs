using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Yolov5Net.Scorer.Extensions;
using Yolov5Net.Scorer.Models.Abstract;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Object detector.
    /// </summary>
    public class YoloScorer<T> where T : YoloModel
    {
        private readonly T _model;

        /// <summary>
        /// Fits input to net format.
        /// </summary>
        private Bitmap ResizeImage(Image image)
        {
            PixelFormat format = image.PixelFormat;

            var result = new Bitmap(_model.Width, _model.Height, format);

            result.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            var rect = new Rectangle(0, 0, _model.Width, _model.Height);

            using (var graphics = Graphics.FromImage(result))
            {
                graphics.DrawImage(image, rect);
            }

            return result;
        }

        /// <summary>
        /// Extracts pixels into tensor for net input.
        /// </summary>
        private Tensor<float> ExtractPixels(Image image)
        {
            var bitmap = new Bitmap(image);

            var rectangle = new Rectangle(0, 0, image.Width, image.Height);

            BitmapData locked = bitmap.LockBits(rectangle, ImageLockMode.ReadOnly, image.PixelFormat);

            var tensor = new DenseTensor<float>(new[] { 1, 3, image.Height, image.Width });

            unsafe // speed up conversion by direct work with memory
            {
                for (int y = 0; y < locked.Height; y++)
                {
                    byte* row = (byte*)locked.Scan0 + (y * locked.Stride);

                    for (int x = 0; x < locked.Width; x++)
                    {
                        tensor[0, 0, y, x] = row[x * 3 + 0] / 255.0F;
                        tensor[0, 1, y, x] = row[x * 3 + 1] / 255.0F;
                        tensor[0, 2, y, x] = row[x * 3 + 2] / 255.0F;
                    }
                }

                bitmap.UnlockBits(locked);
            }

            return tensor;
        }

        /// <summary>
        /// Runs inference session.
        /// </summary>
        private DenseTensor<float> Inference(Image image)
        {
            Bitmap resized = null;

            if (image.Width != _model.Width || image.Height != _model.Height)
            {
                resized = ResizeImage(image); // fit image size to specified input size
            }

            var inference = new InferenceSession(_model.Weights);

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(resized ?? image))
            };

            var result = inference.Run(inputs);

            object output = result.First(x => x.Name == "output").Value;

            return (DenseTensor<float>)output;
        }

        /// <summary>
        /// Parses net output to predictions.
        /// </summary>
        private List<YoloPrediction> ParseOutput(DenseTensor<float> output, Image image)
        {
            var result = new List<YoloPrediction>();

            var (xGain, yGain) = (_model.Width / (float)image.Width, _model.Height / (float)image.Height);

            for (int i = 0; i < output.Length / _model.Dimensions; i++) // iterate tensor
            {
                if (output[0, i, 4] <= _model.Confidence) continue;

                for (int j = 5; j < _model.Dimensions; j++) // compute mul conf
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; // conf = obj_conf * cls_conf
                }

                for (int k = 5; k < _model.Dimensions; k++)
                {
                    if (output[0, i, k] <= _model.MulConfidence) continue;

                    var xMin = (output[0, i, 0] - output[0, i, 2] / 2) / xGain; // top left x
                    var yMin = (output[0, i, 1] - output[0, i, 3] / 2) / yGain; // top left y
                    var xMax = (output[0, i, 0] + output[0, i, 2] / 2) / xGain; // bottom right x
                    var yMax = (output[0, i, 1] + output[0, i, 3] / 2) / yGain; // bottom right y

                    YoloLabel label = _model.Labels[k - 5];

                    var prediction = new YoloPrediction(label, output[0, i, k])
                    {
                        Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                    };

                    result.Add(prediction);
                }
            }

            return result;
        }

        /// <summary>
        /// Removes overlaped duplicates (nms).
        /// </summary>
        private List<YoloPrediction> Supress(List<YoloPrediction> items)
        {
            var result = new List<YoloPrediction>(items);

            foreach (var item in items)
            {
                foreach (var current in result.ToList())
                {
                    if (current == item) continue;

                    var (rect1, rect2) = (item.Rectangle, current.Rectangle);

                    RectangleF intersection = RectangleF.Intersect(rect1, rect2);

                    float intArea = intersection.Area();
                    float unionArea = rect1.Area() + rect2.Area() - intArea;
                    float overlap = intArea / unionArea;

                    if (overlap > _model.Overlap)
                    {
                        if (item.Score > current.Score)
                        {
                            result.Remove(current);
                        }
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Runs object detection.
        /// </summary>
        public List<YoloPrediction> Predict(Image image)
        {
            return Supress(ParseOutput(Inference(image), image));
        }

        public YoloScorer()
        {
            _model = Activator.CreateInstance<T>();
        }
    }
}
