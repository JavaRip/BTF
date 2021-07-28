export interface NeuralNetworkConstructor {
  inputToHiddenWeights: number[][];
  hiddenToOutputWeights: number[][];
}

export interface NeuralNetworkInterface {
  constructor: NeuralNetworkConstructor;
  train: void;
  query: void;
  matrixGenerator: number[][];

  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;
  learningRate: number;
}
