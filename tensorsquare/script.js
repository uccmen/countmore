/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const squaresDataResponse = await fetch('https://raw.githubusercontent.com/uccmen/countmore/master/square-data.json');
  const squaresData = await squaresDataResponse.json();
  console.log('DATA', squaresData);

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
    {name: 'Answer vs Square Of'},
    {values},
    {
      xLabel: 'Answer',
      yLabel: 'Square Of',
      height: 300
    }
  );

  // More code will be added below
}

document.addEventListener('DOMContentLoaded', run);
