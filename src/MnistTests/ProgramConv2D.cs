// Machine Learning
// File name: ProgramConv2D.cs
// Code It Yourself with .NET, 2024

using MachineLearning;
using MachineLearning.Typed.NeuralNetwork;
using MachineLearning.Typed.NeuralNetwork.Layers;
using MachineLearning.Typed.NeuralNetwork.Losses;
using MachineLearning.Typed.NeuralNetwork.Operations;
using MachineLearning.Typed.NeuralNetwork.ParamInitializers;

namespace MnistTests;

internal class MnistConvNeuralNetwork(SeededRandom? random)
    : NeuralNetwork<float[,,,], float[,]>(new SoftmaxCrossEntropyLoss(), random)
{
    protected override LayerBuilder<float[,]> OnAddLayers(LayerBuilder<float[,,,]> builder)
    {
        GlorotInitializer initializer = new(Random);
        Dropout4D? dropout = new(0.85f, Random);

        return builder
            .AddLayer(new Conv2DLayer(
                filters: 16,
                kernelSize: 3,
                activationFunction: new Tanh4D(),
                paramInitializer: initializer,
                dropout: dropout
            ))
            .AddLayer(new FlattenLayer())
            .AddLayer(new DenseLayer(10, new Linear(), initializer));
    }

}

internal class ProgramConv2D
{
    private static void Main(string[] args)
    {

    }
}
