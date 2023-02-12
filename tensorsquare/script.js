/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const squaresDataResponse = await fetch('https://raw.githubusercontent.com/uccmen/countmore/master/square-data.json');
  const squaresData = await squaresDataResponse.json();

  return squaresData;
}


async function run() {
  // Load and plot the original input data that we are going to train on.
  const data = await getData();
  const values = data.map(d => ({
    x: d.squareOf,
    y: d.answer,
  }));

  tfvis.render.scatterplot(
    {name: 'Square Of vs Answer'},
    {values},
    {
      xLabel: 'Square Of',
      yLabel: 'Answer',
      height: 300
    }
  );

  // More code will be added below
  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

// Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
  console.log('Done Testing');
}

function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

  // Add hidden layers with activation function
  model.add(tf.layers.dense({units: 1}));
  model.add(tf.layers.dense({units: 1, activation: 'relu'}));
  model.add(tf.layers.dense({units: 1}));
  model.add(tf.layers.dense({units: 2, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1}));
  model.add(tf.layers.dense({units: 2, activation: 'sigmoid'}));
  model.add(tf.layers.dense({units: 1}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

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

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.squareOf)
    const labels = data.map(d => d.answer);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
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

  const batchSize = 32;
  const epochs = 50;

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

  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xsDecimal, predsDecimal] = tf.tidy(() => {

    const xsNormDecimal = tf.linspace(0, 1, 100);
    const predictionsDecimal = model.predict(xsNormDecimal.reshape([100, 1]));

    const unNormXsDecimal = xsNormDecimal
      .mul(inputMax.sub(inputMin))
      .add(inputMin);

    const unNormPredsDecimal = predictionsDecimal
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormXsDecimal.dataSync(), unNormPredsDecimal.dataSync()];
  });


  const predictedPointsDecimal = Array.from(xsDecimal).map((val, i) => {
    return {x: val, y: predsDecimal[i], actual: val**2}
  });

  const originalPoints = inputData.map(d => ({
    x: d.squareOf, y: d.answer,
  }));


  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data'},
    {values: [originalPoints, predictedPointsDecimal], series: ['original', 'predictedDecimal']},
    {
      xLabel: 'Square Of',
      yLabel: 'Answer',
      height: 300
    }
  );
}

// Create the model
const model = createModel();
tfvis.show.modelSummary({name: 'Model Summary'}, model);

document.addEventListener('DOMContentLoaded', run);
