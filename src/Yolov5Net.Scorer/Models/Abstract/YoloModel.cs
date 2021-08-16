using System.Collections.Generic;
using System.IO;

namespace Yolov5Net.Scorer.Models.Abstract
{
    /// <summary>
    /// Model descriptor.
    /// </summary>
    public abstract class YoloModel
    {
        public abstract int Width { get; set; }
        public abstract int Height { get; set; }
        public abstract int Depth { get; set; }
        public abstract int Dimensions { get; set; }
        public abstract float[] Strides { get; set; }
        public abstract float[][][] Anchors { get; set; }
        public abstract int[] Shapes { get; set; }
        public abstract float Confidence { get; set; }
        public abstract float MulConfidence { get; set; }
        public abstract float Overlap { get; set; }
        public abstract string[] Outputs { get; set; }
        public abstract List<YoloLabel> Labels { get; set; }
        public abstract bool UseDetect { get; set; }

        public byte[] Weights { get; set; }

        public YoloModel(string weights)
        {
            Weights = File.ReadAllBytes(weights);
        }

        public YoloModel(Stream weights)
        {
            using (var reader = new BinaryReader(weights))
            {
                Weights = reader.ReadBytes((int)weights.Length);
            }
        }

        public YoloModel(byte[] weights)
        {
            Weights = weights;
        }
    }
}
