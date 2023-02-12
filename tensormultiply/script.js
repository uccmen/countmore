console.log('Hello TensorFlow');


/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const multiplyTimeTable = await fetch('https://raw.githubusercontent.com/uccmen/countmore/master/multiplication-table.txt');
  const timeTableData = await multiplyTimeTable.text();

  let inputs = [];
  let labels = [];

  let lines = timeTableData.trim().split('\n');

  lines.forEach((line) => {
    let parts = line.split('=');
    let input = parts[0].split('x').map(x => parseInt(x));
    let label = [parseInt(parts[1])];

    inputs.push(input);
    labels.push(label);
  });

  return {
    inputs,
    labels
  };
}


async function run() {
  // Load and plot the original input data that we are going to train on.
  const trainingData = await getData();

  const data = trainingData.inputs.map((input, index) => ({
    x: input[0], y: input[1], result: trainingData.labels[index][0]
  }))

  tfvis.render.scatterplot(
    {name: 'Multiplication Table Example'},
    {values: data, series: 'Result'},
    {
      xLabel: 'X Operand',
      yLabel: 'Y Operand',
      pointSize: 30,
      tooltip: {
        header: 'Result:',
        renderer: (obj) => obj.result
      },
      zoomToFit: true
    }
  );

  // More code will be added below
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(trainingData);
  const {inputs, labels} = tensorData;

// Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');
  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, trainingData, tensorData);
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add two input layer e.g. [1, 2] ~ 1 * 2 = 2
  model.add(tf.layers.dense({inputShape: [2], units: 1, useBias: true}));

  // Add an output layer e.g. [2] ~= 1 * 2
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.


  // convert data into [{input: [1, 2], label: [2]}]
  const d = data.inputs.map((input, index) => ({
    input,
    label: data.labels[index]
  }));

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(d);

    // Step 2. Convert data to Tensor
    const inputs = d.map(data => data.input);
    const labels = d.map(data => data.label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 2]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 64;
  const epochs = 15;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  console.log('INPUT MAX', inputMax);
  console.log('INPUT MIN', inputMin);
  console.log('LABEL MIN', labelMin);
  console.log('LABEL MAX', labelMax);

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {

    const xsNorm = tf.linspace(0, 1, 200).reshape([100, 2]);
    const predictions = model.predict(xsNorm);

    const unNormXs = xsNorm
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXs.floor().arraySync(), unNormPreds.floor().dataSync()];
  });

  const predictedPoints = Array.from(xs).map((input, i) => {
    return {x: input[0], y: input[1], result: preds[i]}
  });

  const originalPoints = inputData.inputs.map((input, index) => ({
    x: input[0], y: input[1], result: inputData.labels[index][0]
  }))


  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPoints], series: ['original', 'predicted']},
    {
      xLabel: 'X Operand',
      yLabel: 'Y Operand',
      height: 300
    }
  );

  console.log("MORE PREDICTION", model.predict(tf.tensor2d([[1, 1], [1, 2], [20, 2000], [2000, 20]])).arraySync())
}

// Create the model
const model = createModel();
tfvis.show.modelSummary({name: 'Model Summary'}, model);


document.addEventListener('DOMContentLoaded', run);
