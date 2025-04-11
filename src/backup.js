import {drawConnectors, drawLandmarks} from '@mediapipe/drawing_utils';
import {HAND_CONNECTIONS, Hands} from "@mediapipe/hands";
import {Camera} from "@mediapipe/camera_utils";

const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('canvas');
const ctx = canvasElement.getContext('2d');
const predictionText = document.getElementById('prediction');
const simonText = document.getElementById('simon');
const startBtton = document.getElementById('start');
const feedbackText = document.getElementById('feedback');

const optionArray = ['fist', 'openHand', 'thumbsUp', 'thumbsDown'];
let orderArray = [];
let answerArray = [];
let level = 1;
let arrayEntry = 0;
let gameStarted = false;
let initialDelay = 1000;
let holdTime = 2000;
let gameLoopCounter = 0

let handLandmarks = []


ml5.setBackend("webgl");
const neuralNetwork = ml5.neuralNetwork({task: 'classification', debug: true})
const modelDetails = {
    model: 'model/model.json',
    metadata: 'model/model_meta.json',
    weights: 'model/model.weights.bin'
}
neuralNetwork.load(modelDetails, () => console.log("het model is geladen!"))
ctx.imageSmoothingEnabled = true;

// Ensure video and canvas have the same resolution
const videoWidth = 640;
const videoHeight = 480;

videoElement.width = videoWidth;
videoElement.height = videoHeight;

canvasElement.width = videoWidth;
canvasElement.height = videoHeight;

const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});

hands.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

startBtton.addEventListener('click', (event) => {
    startGame();
})

function randomIntInRange(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
}

function startGame() {
    if (!gameStarted) {
        gameStarted = true;
        startBtton.innerText = 'Stop Game';

        initLevel();

    } else{
        startBtton.innerText = 'Start Game';
        gameStarted = false;
        orderArray = [];
        answerArray = [];
        level = 1;
        arrayEntry = 0;
        initialDelay = 1000;
        gameLoopCounter = 0;
    }
}

function initLevel(){
    orderArray = [];

    for (let i = 0; i < level + 1; i++) {
        orderArray.push(optionArray[randomIntInRange(0, optionArray.length - 1)]);
    }

    console.log(orderArray);

    orderArray.push('GO!')

    startLevel();
}


function startLevel() {
    setTimeout(function() {
        simonText.innerText = orderArray[gameLoopCounter];
        gameLoopCounter++;
        if (gameLoopCounter < orderArray.length) {
            startLevel();
        }
    }, initialDelay);
}

function drawLandmarksFunc() {
    ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    for (const hand of handLandmarks) {
        drawConnectors(ctx, hand, HAND_CONNECTIONS, {
            color: 'black',
            lineWidth: 2
        });

        drawLandmarks(ctx, hand, {
            color: 'red',
            lineWidth: 1
        });
    }
}

hands.onResults((results) => {
    handLandmarks = results.multiHandLandmarks || [];
    drawLandmarksFunc();
    normalizeAndDetectHandData()
});

// Start camera feed
const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: videoWidth,
    height: videoHeight
});

function normalizeAndDetectHandData() {
    const flatHandArray = [];

    if (handLandmarks[0] && gameStarted) {
        if (handLandmarks[0].length > 0) {
            const wrist = handLandmarks[0][0]; // Wrist is always index 0

            for (let i = 0; i < handLandmarks[0].length; i++) {
                const landmark = handLandmarks[0][i];

                const x = landmark.x - wrist.x;
                const y = landmark.y - wrist.y;
                const z = landmark.z - wrist.z;

                flatHandArray.push(x, y, z);
            }

            makePrediction(flatHandArray);

            return flatHandArray;
        }

    } else if(gameStarted) {
        predictionText.innerHTML = "No Hands Detected"
        return null;
    } else{
        predictionText.innerHTML = 'Start Game To Start Detecting Gestures!'
        return null;
    }
    return null;
}

async function makePrediction(data) {
    const results = await neuralNetwork.classify(data) // dit is een pose uit mediapipe

    let highestConfidence = results[0]
    let label

// Loop through the predictions array and find the one with the highest confidence
    for (let i = 1; i < results.length; i++) {
        if (results[i].confidence > highestConfidence.confidence) {
            highestConfidence = results[i];
        }
    }

    if (highestConfidence.confidence > 0.8) {
        predictionText.innerHTML = highestConfidence.label.slice(0, -1)

        if ( highestConfidence.label.slice(0, -1) === orderArray[arrayEntry]){
            answerArray.push('correct')
            console.log('correct')
        }  else{
            answerArray.push('false')
            console.log('false')
        }

        if(answerArray.length > 50){
            checkAnswers();
        }

    } else {
        predictionText.innerHTML = "Can't detect a gesture"
    }
}

function checkAnswers() {

    const correctCount = answerArray.filter(result => result === 'correct').length;
    const falseCount = answerArray.filter(result => result === 'false').length;
    const total = correctCount + falseCount;

    if (total === 0) {
        console.log("No data to evaluate.");
        return;
    }

    const accuracy = (correctCount / total) * 100;

    console.log(`Accuracy: ${accuracy.toFixed(2)}%`);

    if (accuracy > 80) {
        console.log("🎉 Accuracy is above 80%! Something awesome happens!");
        // Trigger your event here
        feedbackText.innerHTML = 'Correct!'
        arrayEntry++
        answerArray = [];
    } else{
        feedbackText.innerHTML = 'Incorrect, Game Over :('
        startGame();
    }
}


camera.start();
