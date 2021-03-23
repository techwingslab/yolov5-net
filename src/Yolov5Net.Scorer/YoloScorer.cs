using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
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

        private readonly float[] _strides = new float[] { 8, 16, 32 };

        private readonly float[][][] _anchors = new float[][][]
        {
            new float[][] { new float[] { 010, 13 }, new float[] { 016, 030 }, new float[] { 033, 023 } },
            new float[][] { new float[] { 030, 61 }, new float[] { 062, 045 }, new float[] { 059, 119 } },
            new float[][] { new float[] { 116, 90 }, new float[] { 156, 198 }, new float[] { 373, 326 } }
        };

        private readonly int[] _shapes = new int[] { 80, 40, 20 };

        /// <summary>
        /// Outputs value between 0 and 1.
        /// </summary>
        private float Sigmoid(float value)
        {
            return 1 / (1 + MathF.Exp(-value));
        }

        /// <summary>
        /// Converts xywh bbox format to xyxy.
        /// </summary>
        private float[] Xywh2xyxy(float[] source)
        {
            var result = new float[4];

            result[0] = source[0] - source[2] / 2f;
            result[1] = source[1] - source[3] / 2f;
            result[2] = source[0] + source[2] / 2f;
            result[3] = source[1] + source[3] / 2f;

            return result;
        }

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
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBilinear;
                graphics.SmoothingMode = SmoothingMode.HighQuality;

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
        private DenseTensor<float>[] Inference(Image image)
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

            DenseTensor<float>[] output = new[]
            {
                result.First(x => x.Name == "output1").Value as DenseTensor<float>,
                result.First(x => x.Name == "output2").Value as DenseTensor<float>,
                result.First(x => x.Name == "output3").Value as DenseTensor<float>
            };

            return output;
        }

        /// <summary>
        /// Parses net output to predictions.
        /// </summary>
        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            var result = new List<YoloPrediction>();

            var (xGain, yGain) = (_model.Width / (float)image.Width, _model.Height / (float)image.Height);

            for (int i = 0; i < output.Length; i++) // iterate outputs
            {
                int shapes = _shapes[i]; // shapes per output

                for (int a = 0; a < _anchors.Length; a++) // iterate anchors
                {
                    for (int y = 0; y < shapes; y++) // iterate rows
                    {
                        for (int x = 0; x < shapes; x++) // iterate columns
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            var objConfidence = buffer[4]; // extract object confidence

                            if (objConfidence < _model.Confidence) continue; // skip low confidence objects

                            List<float> scores = buffer.Skip(5).Select(x => x * objConfidence).ToList();

                            float mulConfidence = scores.Max(); // find the best label score

                            if (mulConfidence <= _model.MulConfidence) continue; // skip if no any satisfied class

                            var rawX = (buffer[0] * 2 - 0.5f + x) * _strides[i]; // bbox x (center)
                            var rawY = (buffer[1] * 2 - 0.5f + y) * _strides[i]; // bbox y (center)
                            var rawW = MathF.Pow(buffer[2] * 2, 2) * _anchors[i][a][0]; // bbox width
                            var rawH = MathF.Pow(buffer[3] * 2, 2) * _anchors[i][a][1]; // bbox height

                            float[] xyxy = Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            var xMin = xyxy[0] / xGain; // bbox top left corner x
                            var yMin = xyxy[1] / yGain; // bbox top left corner y
                            var xMax = xyxy[2] / xGain; // bbox bottom right corner x
                            var yMax = xyxy[3] / yGain; // bbox bottom right corner y

                            YoloLabel label = _model.Labels[scores.IndexOf(mulConfidence)];

                            var prediction = new YoloPrediction(label, mulConfidence)
                            {
                                Rectangle = new RectangleF(xMin, yMin, xMax - xMin, yMax - yMin)
                            };

                            result.Add(prediction);
                        }
                    }
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
