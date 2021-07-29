class NeuralNetwork {
  inputNodes: number;
  hiddenNodes: number;
  outputNodes: number;
  learningRate: number;
  inputToHiddenWeights: number[][];
  hiddenToOutputWeights: number[][];

  constructor(inputNodes: number, hiddenNodes: number, outputNodes: number, learningRate: number) {
    // initial attributes
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.outputNodes = outputNodes;
    this.learningRate = learningRate;

    // calculate random starting weight matrices
    this.inputToHiddenWeights = this.matrixGenerator(this.hiddenNodes, this.inputNodes);
    this.hiddenToOutputWeights = this.matrixGenerator(this.outputNodes, this.hiddenNodes);
  }

  train() {
    return '🏋️';
  }

  query() {
    return '🧐';
  }

  matrixGenerator(rows: number, cols: number) {
    const matrix: number[][] = [];
    for (let i = 0; i < cols; i += 1) {
      const columns: number[] = [];
      for (let j = 0; j < rows; j += 1) {
        columns.push(Math.random());
      }
      matrix.push(columns);
    }
    return matrix;
  }
}

function hello(meme: string) {
  console.log(meme);
  return 'string';
}

const neuralNetwork = new NeuralNetwork(3, 3, 3, 3);
console.log(neuralNetwork);
