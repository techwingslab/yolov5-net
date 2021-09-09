using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using Yolov5Net.Scorer.Extensions;
using Yolov5Net.Scorer.Models.Abstract;

namespace Yolov5Net.Scorer
{
    /// <summary>
    /// Yolov5 scorer.
    /// </summary>
    public class YoloScorer<T> : IDisposable where T : YoloModel
    {
        private readonly T _model;

        private readonly InferenceSession _inferenceSession;

        /// <summary>
        /// Outputs value between 0 and 1.
        /// </summary>
        private float Sigmoid(float value)
        {
            return 1 / (1 + (float)Math.Exp(-value));
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
        /// Returns value clamped to the inclusive range of min and max.
        /// </summary>
        public float Clamp(float value, float min, float max)
        {
            return (value < min) ? min : (value > max) ? max : value;
        }

        /// <summary>
        /// Resizes image keeping ratio to fit model input size.
        /// </summary>
        private Bitmap ResizeImage(Image image)
        {
            PixelFormat format = image.PixelFormat;

            var output = new Bitmap(_model.Width, _model.Height, format);

            using (var graphics = Graphics.FromImage(output))
            {
                graphics.Clear(Color.FromArgb(0, 0, 0, 0)); // ckear with black

                var (wRatio, hRatio) = (_model.Width / (float)image.Width, _model.Height / (float)image.Height);

                var ratio = Math.Min(wRatio, hRatio); // min ratio = resized / original

                var (width, height) = ((int)(image.Width * ratio), (int)(image.Height * ratio));

                var (x, y) = ((_model.Width / 2) - (width / 2), (_model.Height / 2) - (height / 2));

                graphics.DrawImage(image, new Rectangle(x, y, width, height));
            }

            return output;
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

            var inputs = new List<NamedOnnxValue> // add image as onnx input
            {
                NamedOnnxValue.CreateFromTensor("images", ExtractPixels(resized ?? image))
            };

            var result = _inferenceSession.Run(inputs); // run inference session

            var output = new List<DenseTensor<float>>();

            foreach (var item in _model.Outputs) // add outputs for processing
            {
                output.Add(result.First(x => x.Name == item).Value as DenseTensor<float>);
            };

            return output.ToArray();
        }

        /// <summary>
        /// Parses net output (detect) to predictions.
        /// </summary>
        private List<YoloPrediction> ParseDetect(DenseTensor<float> output, Image image)
        {
            var result = new List<YoloPrediction>();

            var (xGain, yGain) = (_model.Width / (float)image.Width, _model.Height / (float)image.Height);

            var gain = Math.Min(xGain, yGain); // min gain  = resized / original

            var (xPad, yPad) = ((_model.Width - image.Width * gain) / 2, (_model.Height - image.Height * gain) / 2);

            for (int i = 0; i < output.Length / _model.Dimensions; i++) // iterate tensor
            {
                if (output[0, i, 4] <= _model.Confidence) // skip low confidence
                    continue;

                for (int j = 5; j < _model.Dimensions; j++)
                {
                    output[0, i, j] = output[0, i, j] * output[0, i, 4]; // mul conf = obj_conf * cls_conf
                }

                for (int k = 5; k < _model.Dimensions; k++)
                {
                    if (output[0, i, k] <= _model.MulConfidence) // skip low confidence
                        continue;

                    var xMin = ((output[0, i, 0] - output[0, i, 2] / 2) - xPad) / gain; // unpad bbox tlx
                    var yMin = ((output[0, i, 1] - output[0, i, 3] / 2) - yPad) / gain; // unpad bbox tly
                    var xMax = ((output[0, i, 0] + output[0, i, 2] / 2) - xPad) / gain; // unpad bbox brx
                    var yMax = ((output[0, i, 1] + output[0, i, 3] / 2) - yPad) / gain; // unpad bbox bry

                    xMin = Clamp(xMin, 0, image.Width);  // clip bbox tlx to boundaries
                    yMin = Clamp(yMin, 0, image.Height); // clip bbox tly to boundaries
                    xMax = Clamp(xMax, 0, image.Width);  // clip bbox brx to boundaries
                    yMax = Clamp(yMax, 0, image.Height); // clip bbox bry to boundaries

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
        /// Parses net outputs (sigmoid) to predictions.
        /// </summary>
        private List<YoloPrediction> ParseSigmoid(DenseTensor<float>[] output, Image image)
        {
            var result = new List<YoloPrediction>();

            var (xGain, yGain) = (_model.Width / (float)image.Width, _model.Height / (float)image.Height);

            var gain = Math.Min(xGain, yGain); // min gain  = resized / original

            var (xPad, yPad) = ((_model.Width - image.Width * gain) / 2, (_model.Height - image.Height * gain) / 2);

            for (int i = 0; i < output.Length; i++) // iterate outputs
            {
                int shapes = _model.Shapes[i]; // shapes per output

                for (int a = 0; a < _model.Anchors.Length; a++) // iterate anchors
                {
                    for (int y = 0; y < shapes; y++) // iterate rows
                    {
                        for (int x = 0; x < shapes; x++) // iterate columns
                        {
                            int offset = (shapes * shapes * a + shapes * y + x) * _model.Dimensions;

                            float[] buffer = output[i].Skip(offset).Take(_model.Dimensions).Select(Sigmoid).ToArray();

                            var objConfidence = buffer[4]; // extract object confidence

                            if (objConfidence < _model.Confidence) // check predicted object confidence
                                continue;

                            List<float> scores = buffer.Skip(5).Select(b => b * objConfidence).ToList();

                            float mulConfidence = scores.Max(); // find the best label

                            if (mulConfidence <= _model.MulConfidence) // check class obj_conf * cls_conf confidence
                                continue;

                            var rawX = (buffer[0] * 2 - 0.5f + x) * _model.Strides[i]; // predicted bbox x (center)
                            var rawY = (buffer[1] * 2 - 0.5f + y) * _model.Strides[i]; // predicted bbox y (center)

                            var rawW = (float)Math.Pow(buffer[2] * 2, 2) * _model.Anchors[i][a][0]; // predicted bbox width
                            var rawH = (float)Math.Pow(buffer[3] * 2, 2) * _model.Anchors[i][a][1]; // predicted bbox height

                            float[] xyxy = Xywh2xyxy(new float[] { rawX, rawY, rawW, rawH });

                            var xMin = Clamp((xyxy[0] - xPad) / gain, 0, image.Width);  // unpad and clip bbox tlx
                            var yMin = Clamp((xyxy[1] - yPad) / gain, 0, image.Height); // unpad and clip bbox tly
                            var xMax = Clamp((xyxy[2] - xPad) / gain, 0, image.Width);  // unpad and clip bbox brx
                            var yMax = Clamp((xyxy[3] - yPad) / gain, 0, image.Height); // unpad and clip bbox bry

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
        /// Parses net outputs (sigmoid or detect layer) to predictions.
        /// </summary>
        private List<YoloPrediction> ParseOutput(DenseTensor<float>[] output, Image image)
        {
            return _model.UseDetect ? ParseDetect(output[0], image) : ParseSigmoid(output, image);
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

        /// <summary>
        /// Creates new instance of YoloScorer.
        /// </summary>
        public YoloScorer()
        {
            _model = Activator.CreateInstance<T>();
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights path and options.
        /// </summary>
        public YoloScorer(string weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(File.ReadAllBytes(weights), opts ?? new SessionOptions());
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights stream and options.
        /// </summary>
        public YoloScorer(Stream weights, SessionOptions opts = null) : this()
        {
            using (var reader = new BinaryReader(weights))
            {
                _inferenceSession = new InferenceSession(reader.ReadBytes((int)weights.Length), opts ?? new SessionOptions());
            }
        }

        /// <summary>
        /// Creates new instance of YoloScorer with weights bytes and options.
        /// </summary>
        public YoloScorer(byte[] weights, SessionOptions opts = null) : this()
        {
            _inferenceSession = new InferenceSession(weights, opts ?? new SessionOptions());
        }

        /// <summary>
        /// Disposes YoloScorer instance.
        /// </summary>
        public void Dispose()
        {
            _inferenceSession.Dispose();
        }
    }
}
