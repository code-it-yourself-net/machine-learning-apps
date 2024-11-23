// Machine Learning
// File name: Program.cs
// Code It Yourself with .NET, 2024

using System.Diagnostics;
using System.Threading.Channels;

using MachineLearning;
using MachineLearning.NeuralNetwork.LearningRates;
using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Losses;
using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.Optimizers;
using MachineLearning.Typed;

using static MachineLearning.Typed.ArrayUtils;

// Prepare data:
// 3 (none, vertical, and horizontal stripes) * 4 examples * 1 channel * 3 rows * 3 columns

const int inputChannels = 1;
const int examples = 15;
const int outputCategories = 3; // 0, 1, 2

float[,,,] xTrain = new float[examples, inputChannels, 3, 3]
{
    // neither vertical nor horizonta
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 1, 0, 1 },
            { 0, 1, 0 },
            { 1, 0, 1 }
        }
    },
    {
        {
            { 0, 1, 0 },
            { 1, 1, 1 },
            { 0, 1, 0 },
        }
    },
    {
        {
            { 1, 0, 0 },
            { 0, 1, 0 },
            { 0, 0, 1 },
        }
    },
    // horizontal
    {
        {
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 1, 1, 1 },
            { 0, 0, 0 },
            { 0, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 0, 0, 0 },
            { 1, 1, 1 }
        }
    },
    {
        {
            { 0, 0, 0 },
            { 1, 1, 1 },
            { 0, 0, 0 },
        }
    },
    {
        {
            { 1, 1, 1 },
            { 1, 1, 1 },
            { 0, 0, 0 },
        }
    },
    // vertical
    {
        {
            { 1, 0, 1 },
            { 1, 0, 1 },
            { 1, 0, 1 }
        }
    },
    {
        {
            { 0, 1, 0 },
            { 0, 1, 0 },
            { 0, 1, 0 }
        }
    },
    {
        {
            { 1, 0, 0 },
            { 1, 0, 0 },
            { 1, 0, 0 }
        }
    },
    {
        {
            { 0, 0, 1 },
            { 0, 0, 1 },
            { 0, 0, 1 },
        }
    },
    {
        {
            { 0, 1, 1 },
            { 0, 1, 1 },
            { 0, 1, 1 },
        }
    }
};

// Values - 0 for unknown, 1 for horizontal stripes, 2 for vertical stripes
float[,] oneHot = new float[examples, outputCategories] {
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 1, 0 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
    { 0, 0, 1 },
};

float loss = Train(xTrain, oneHot, 1_000, 241123);

Console.WriteLine($"loss: {loss}");
Console.ReadLine();

static float Train(float[,,,] xTrain, float[,] yTrain, int iterations = 2_000 /*, float learningRate = 0.01f */, int? seed = null)
{
    float loss = 0;
    Random random;
    if (seed.HasValue)
        random = new(seed.Value);
    else
        random = new();

    const int outputChannels = 16;
    float[,,,] kernels = CreateRandom(inputChannels, outputChannels, 3, 3, random);
    Conv2D conv2D = new(kernels);
    Tanh4D tanh4D = new();
    Flatten flatten = new();

    float[,] weights = CreateRandom(outputChannels * 3 * 3, outputCategories, random);
    WeightMultiply weightMultiply = new(weights);

    float[] bias = CreateRandom(outputCategories, random);
    BiasAdd biasAdd = new(bias);

    SoftmaxCrossEntropyLoss softmaxCrossEntropyLoss = new();

    LearningRate learningRate = new ExponentialDecayLearningRate(0.19f, 0.05f);
    Optimizer optimizer = new StochasticGradientDescent(learningRate);

    for (int i = 0; i < iterations; i++)
    {
        (xTrain, yTrain) = PermuteData(xTrain, yTrain, random);

        // Forward

        float[,,,] conv2DOutput = conv2D.Forward(xTrain, false);
        float[,,,] tanh4DOutput = tanh4D.Forward(conv2DOutput, false);
        float[,] flattenOutput = flatten.Forward(tanh4DOutput, false);
        float[,] weightMultiplyOutput = weightMultiply.Forward(flattenOutput, false);
        float[,] biasAddOutput = biasAdd.Forward(weightMultiplyOutput, false);

        loss = softmaxCrossEntropyLoss.Forward(biasAddOutput, yTrain);

        // Print loss every 10 steps

        if (i % 10 == 0)
        {
            Console.WriteLine($"iteration: {i}, loss: {loss}");

            // Write the percent of the correct predictions
            float[,] prediction = softmaxCrossEntropyLoss.Prediction;
            int[] predictionArgmax = prediction.Argmax();
            int rows = predictionArgmax.GetLength(0);
            int hits = 0;
            for (int row = 0; row < rows; row++)
            {
                int predictedDigit = predictionArgmax[row];
                if (yTrain[row, predictedDigit] == 1f)
                    hits++;
            }
            float accuracy = (float)hits / rows;
            Console.WriteLine($"accuracy: {accuracy}");
        }

        // Backward

        float[,] softmaxCrossEntropyLossGradient = softmaxCrossEntropyLoss.Backward();

        float[,] biasAddGradient = biasAdd.Backward(softmaxCrossEntropyLossGradient);
        float[,] weightMultiplyGradient = weightMultiply.Backward(biasAddGradient);
        float[,,,] flattenGradient = flatten.Backward(weightMultiplyGradient);
        float[,,,] tanh4DGradient = tanh4D.Backward(flattenGradient);
        float[,,,] conv2DGradient = conv2D.Backward(tanh4DGradient);

        // Update weights

        optimizer.UpdateLearningRate(i + 1, iterations);
        conv2D.UpdateParams(null, optimizer);
        weightMultiply.UpdateParams(null, optimizer);
        biasAdd.UpdateParams(null, optimizer);
    }

    return loss;
}

