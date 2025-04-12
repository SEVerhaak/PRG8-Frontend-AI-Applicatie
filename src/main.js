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
const fist = document.getElementById('fist');
const thumbsDown = document.getElementById('thumbsDown')
const thumbsUp = document.getElementById('thumbsUp')
const openHand = document.getElementById('openHand');
const gameElements = [fist, thumbsDown, thumbsUp, openHand];


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

let playerTurn = false;
let levelInProgress = false;
let waitingBetweenGestures = false;
let simonSaysActive = false;


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
        level = 1;
        feedbackText.innerHTML = '';
        initLevel();
    } else {
        resetGame();
    }
}

function resetGame() {
    gameStarted = false;
    startBtton.innerText = 'Start Game';
    orderArray = [];
    answerArray = [];
    level = 1;
    arrayEntry = 0;
    gameLoopCounter = 0;
    simonText.innerText = '-';
    feedbackText.innerText = '';
    predictionText.innerText = 'Start Game To Start Detecting Gestures!';
    levelText.innerText = `Level: ${level}`;
    playerTurn = false;
    levelInProgress = false;
}

function initLevel() {
    orderArray = [];
    let previousGesture = null;

    for (let i = 0; i < level; i++) {
        let nextGesture;
        do {
            nextGesture = optionArray[randomIntInRange(0, optionArray.length)];
        } while (nextGesture === previousGesture);

        orderArray.push(nextGesture);
        previousGesture = nextGesture;
    }

    console.log("Simon Says:", orderArray);
    showInstructions();
}



function showInstructions() {
    levelInProgress = true;
    simonSaysActive = true;  // Pauses detection during this phase
    simonText.innerText = '-';
    gameLoopCounter = 0;

    for (let i = 0; i < gameElements.length; i++) {
        gameElements[i].classList.remove('active');
    }

    const showNext = () => {
        if (gameLoopCounter < orderArray.length) {
            for (let i = 0; i < gameElements.length; i++) {
                gameElements[i].classList.remove('active');
            }

            const element = document.getElementById(orderArray[gameLoopCounter]);
            element.classList.add('active');

            simonText.innerText = orderArray[gameLoopCounter];
            gameLoopCounter++;
            setTimeout(showNext, holdTime);
        } else {
            for (let i = 0; i < gameElements.length; i++) {
                gameElements[i].classList.remove('active');
            }
            simonText.innerText = 'Your turn!';
            arrayEntry = 0;
            answerArray = [];
            playerTurn = true;
            levelInProgress = false;
            simonSaysActive = false;  // Resumes detection
        }
    };

    setTimeout(showNext, initialDelay);
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

const predictionBuffer = [];
const bufferSize = 30;
const confidenceThreshold = 0.8;
const matchThreshold = 0.8; // 80%

async function makePrediction(data) {
    if (waitingBetweenGestures || simonSaysActive) return;  // Don't detect during "Simon Says"

    const results = await neuralNetwork.classify(data);
    let highest = results.reduce((a, b) => a.confidence > b.confidence ? a : b);

    if (highest.confidence > confidenceThreshold) {
        const gesture = highest.label.slice(0, -1); // Normalize to remove L/R
        predictionText.innerText = gesture;

        if (playerTurn && !levelInProgress) {
            predictionBuffer.push(gesture);

            if (predictionBuffer.length >= bufferSize) {
                const expectedGesture = orderArray[arrayEntry];
                const matches = predictionBuffer.filter(g => g === expectedGesture).length;
                const matchRatio = matches / bufferSize;

                if (matchRatio >= matchThreshold) {
                    feedbackText.style.color = 'green';
                    feedbackText.innerText = 'Correct! Next one in 2 seconds...';
                    predictionBuffer.length = 0;
                    waitingBetweenGestures = true;

                    arrayEntry++;

                    // Check if level is completed
                    if (arrayEntry >= orderArray.length) {
                        playerTurn = false;
                        setTimeout(() => {
                            level++;
                            feedbackText.innerText = '';
                            levelText.innerText = `Level: ${level}`;
                            simonText.innerText = '-';
                            waitingBetweenGestures = false;
                            setTimeout(initLevel, 1500);
                        }, 2000);
                    } else {
                        // Wait 2 seconds before allowing next prediction
                        setTimeout(() => {
                            if (gameStarted){
                                feedbackText.innerText = 'Go!';
                                waitingBetweenGestures = false;
                            }
                        }, 2000);
                    }
                } else {
                    feedbackText.style.color = 'red';
                    feedbackText.innerText = 'Wrong gesture! Game Over!';
                    setTimeout(resetGame, 2000);
                }
            }
        }
    } else {
        predictionText.innerText = "Can't detect a gesture";
    }
}


camera.start();