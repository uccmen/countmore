/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
  const squaresDataResponse = await fetch('https://raw.githubusercontent.com/uccmen/countmore/master/square-data-6-16.json');
  const squaresData = await squaresDataResponse.json();
  squaresData.map(squareData => {
    if (squareData.squareOf < 6) {
      console.log('SQUARE OF IS LESS THAN 6', squareData.squareOf);
    }
  });

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
    {name: 'SquareOf v Answer'},
    {values},
    {
      xLabel: 'Square',
      yLabel: 'Answer',
      height: 300
    }
  );

  // More code will be added below
}

document.addEventListener('DOMContentLoaded', run);
