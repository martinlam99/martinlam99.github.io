// LSGAN demo: runs the exported generator (models/generator.onnx) in the browser.
// onnxruntime-web and the model are only downloaded on the first click.
(function () {
  var ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
  var MODEL_URL = 'models/generator.onnx';
  var LATENT_DIM = 128; // must match hyper_parameters.latent_dim in the checkpoint

  var button = document.getElementById('gan-generate');
  var canvas = document.getElementById('gan-canvas');
  var status = document.getElementById('gan-status');
  if (!button || !canvas) return;

  var session = null;

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = function () { reject(new Error('Could not load ' + src)); };
      document.head.appendChild(script);
    });
  }

  async function getSession() {
    if (session) return session;
    status.textContent = 'Loading model…';
    await loadScript(ORT_URL + 'ort.wasm.min.js');
    ort.env.wasm.wasmPaths = ORT_URL;
    ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated, so no threads
    session = await ort.InferenceSession.create(MODEL_URL);
    return session;
  }

  // Standard normal noise, same distribution as torch.randn (Box-Muller)
  function randomLatent() {
    var z = new Float32Array(LATENT_DIM);
    for (var i = 0; i < LATENT_DIM; i += 2) {
      var r = Math.sqrt(-2 * Math.log(1 - Math.random()));
      var theta = 2 * Math.PI * Math.random();
      z[i] = r * Math.cos(theta);
      if (i + 1 < LATENT_DIM) z[i + 1] = r * Math.sin(theta);
    }
    return z;
  }

  // Draws a [1, C, H, W] tensor with values in [-1, 1] (tanh output) onto the canvas
  function draw(tensor) {
    var channels = tensor.dims[1], height = tensor.dims[2], width = tensor.dims[3];
    var data = tensor.data, plane = height * width;
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    var image = ctx.createImageData(width, height);
    for (var p = 0; p < plane; p++) {
      for (var c = 0; c < 3; c++) {
        var value = data[(channels === 1 ? 0 : c) * plane + p];
        image.data[p * 4 + c] = Math.round((value + 1) * 127.5);
      }
      image.data[p * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  button.addEventListener('click', async function () {
    button.disabled = true;
    try {
      var model = await getSession();
      var z = new ort.Tensor('float32', randomLatent(), [1, LATENT_DIM]);
      var output = await model.run({ z: z });
      draw(output.image);
      canvas.classList.add('has-image');
      status.textContent = 'Generated from random noise in your browser.';
    } catch (err) {
      console.error(err);
      status.textContent = 'Could not run the model. Try reloading the page.';
    } finally {
      button.disabled = false;
    }
  });
})();
