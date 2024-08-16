let model;
const imageElement = document.getElementById('uploadedImage');

document.getElementById('loadButton').addEventListener('click', () => {
    const fileInput = document.getElementById('upload');
    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = (e) => {
        imageElement.src = e.target.result;
        imageElement.style.display = 'block';
    };

    if (file) {
        reader.readAsDataURL(file);
    } else {
        alert('Please select an image file to upload.');
    }
});

document.getElementById('identifyButton').addEventListener('click', async () => {
    if (!model) {
        try {
           model = await tf.loadLayersModel('model.json');
        } catch (error) {
            console.error('Error loading model:', error);
            alert('An error occurred while loading the model.');
            return;
        }
    }
    if (imageElement.src) {
        try {
            const inputTensor = preprocessImage(imageElement);
            const prediction = model.predict(inputTensor);
            const predictionData = await prediction.data();
            const disease = getDiseaseName(predictionData);
            document.getElementById('result').textContent = `Disease Identified: ${disease}`;
        } catch (error) {
            console.error('Error during prediction:', error);
            alert('An error occurred during the prediction process.');
        }
    } else {
        alert('Please load an image first.');
    }
});

function preprocessImage(image) {
    const tensor = tf.browser.fromPixels(image)
        .resizeBilinear([224, 224])
        .toFloat()
        .div(tf.scalar(255.0)); // Normalize the image
    return tensor.expandDims(0);
}

function getDiseaseName(predictionData) {
    const classes = ['Healthy', 'Late Blight', 'Early Blight'];
    const maxIndex = predictionData.indexOf(Math.max(...predictionData));
    return classes[maxIndex];
}
