document.addEventListener("DOMContentLoaded", function () {
    // get the game/demo mode
    function getQueryParam(param) {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get(param);
    }

    const gameMode = getQueryParam("mode");

    let currentClickCount = 0;
    const level = 3;
    let correctAnswers = 0;

    // Score board
    const scoreBoard = document.querySelector(".score-container");
    const patchMatchScore = document.getElementById("PatchMatch")
    const userScore = document.getElementById("user");

    // All questions
    let questions = [];

    // Current question
    let currentQuestionIndex = 0;
    let modelCorrectAnswers = 0;
    let correctAnswer = "answer1";
    let modelCorrect = null;
    const images = document.querySelectorAll(".answer-image");
    const imageContainers = document.querySelectorAll(".answer-image-container");
    const questionImage = document.getElementById("PS_tile")
    const feedbackText = document.getElementById("user-feedback-text");


    // PatchMatch prediction
    const patchMatchButton = document.getElementById("PatchMatch-button")
    let patchMatchClicked;
    let similarityText = document.querySelectorAll(".answer-overlay-similarity");
    let rankText = document.querySelectorAll(".answer-overlay-rank")
    const modelPrediction = document.getElementById("prediction-status")
    let numQuestions = 5;

    // Buttons
    let feedbackBox = document.getElementById("feedback-box");
    const nextButtonHTML = document.getElementById('next-button').outerHTML;

    loadQuestions().then(() => {
        loadQuestion(currentQuestionIndex);

        // buttons
        patchMatchButton.addEventListener("click", handlePatchMatchPredictionClick);
        addRestartNextButtons();

        if (gameMode === "demo") {
            numQuestions = questions.length;
        }
    });

    function loadQuestions() {
        return new Promise((resolve, reject) => {
            fetch("questions.json")
                .then(response => response.json())
                .then(data => {
                    questions = data; // Set the questions data
                    resolve();
                })
                .catch(error => {
                    console.error("Error loading questions:", error);
                    reject(error);
                });
        });
    }


    function loadQuestion(questionIndex) {
        const question = questions[questionIndex];

        // Get all S2 tiles
        images.forEach((img, index) => {
            img.src = question.images[index].src;
        });

        // Get model ranks
        rankText.forEach((text, index) => {
            text.textContent = question.images[index].topk;
        });

        // Get model similarites
        similarityText.forEach((text, index) => {
            text.textContent = `Predicted similarity: ${question.images[index].similarity}`;
        });

        // Get PS question tile
        questionImage.src = question.questionImage.src;

        // Get correct answer
        const correctIndex = question.images.findIndex(img => img.correct);
        correctAnswer = correctIndex !== -1 ? `answer${correctIndex + 1}` : null;

        // Get if model was correct
        modelCorrect = question.questionImage.modelCorrect;
        if (modelCorrect) {
            // Increase model's score
            modelCorrectAnswers++;
        }

        // Make images clickable
        if (gameMode === "play") {
            console.log("Running in PLAY mode...");
            makeImagesClickable(true);
            showRanksSimilarities(false);
            showUserFeedback(false);
            showPatchMatchCorrect(false);
            patchMatchClicked = false;

        } else {
            console.log("Running in DEMO mode...");
            if (scoreBoard) scoreBoard.classList.add("hidden");
            showUserFeedback(true);
            makeImagesClickable(false);
            showPatchMatchCorrect(true);
            patchMatchClicked = true;
            showRanksSimilarities(true);
        }

    }

    function endTurn() {
        makeImagesClickable(false);
        showPatchMatchCorrect(true);
        updateScores();
    }

    function handleImageClick(event) {
        currentClickCount++;
        showUserFeedback(true);

        let clickedImage = event.currentTarget;
        if (clickedImage.id === correctAnswer) {
            // Increase user score
            correctAnswers++;

            // Update feedback
            feedbackText.style.color = 'green'
            feedbackText.textContent = 'You are Correct!'

            // Highlight the correct image
            document.getElementById(correctAnswer).style.background = "green";

            // End the turn
            endTurn();
        } else {
            // Update feedback
            feedbackText.style.color = 'red'
            clickedImage.style.background = "red";
            feedbackText.textContent = 'Incorrect!'
        }

        if (currentClickCount === level) {
            // Update feedback
            feedbackText.textContent = 'You are Incorrect!'

            // Highlight the correct answer
            document.getElementById(correctAnswer).style.background = "green";

            // End the turn
            endTurn();
        }
    }

    function updateScores() {
        userScore.textContent = correctAnswers;
        patchMatchScore.textContent = modelCorrectAnswers;
    }

    function showPatchMatchCorrect(show = true) {
        if (show) {
            modelPrediction.style.visibility = "visible";
            if (modelCorrect) {
                modelPrediction.style.color = "green";
                modelPrediction.textContent = "PatchMatch is Correct!";
            } else {
                modelPrediction.style.color = "red";
                modelPrediction.textContent = "PatchMatch is Incorrect";
            }
        } else {
            if (gameMode === 'play') {
                feedbackBox.style.visibility = "hidden";
                modelPrediction.style.visibility = "hidden";
            }
        }
    }

    function handlePatchMatchPredictionClick() {
        if (patchMatchClicked) {
            patchMatchClicked = false;
            showRanksSimilarities(false)
        } else {
            endTurn()
            showUserFeedback(true);
            // Show model prediction ranks (overlays)
            showRanksSimilarities(true)
            patchMatchClicked = true;
        }

    }

    function showRanksSimilarities(show = true) {
        if (show) {
            console.log(correctAnswer)
            imageContainers.forEach((container) => {
                container.classList.add("clicked");
                container.style.pointerEvents = "none";
                if (container.id === correctAnswer) {
                    container.style.background = "green";
                } else {
                    container.style.background = "red";
                }
            });
        } else {
            imageContainers.forEach(container => {
                container.style.background = "none";
                container.classList.remove("clicked");
            });
        }
    }

    function showUserFeedback(show) {
        if (show) {
            if (gameMode === "demo") {
                feedbackText.classList.add("hidden")
            }
            feedbackBox.style.visibility = "visible"
            feedbackBox.style.opacity = "1";
        } else {
            feedbackBox.style.visibility = "hidden";
        }
    }

    function restart() {
        window.location.href = "main.html"
    }

    function addRestartNextButtons() {
        const oldNextButtons = document.getElementById("next-button");
        if (oldNextButtons) {
            oldNextButtons.remove();
        }

        document.querySelector('.controls-container').innerHTML += nextButtonHTML;

        const nextButton = document.getElementById("next-button");
        nextButton.addEventListener("click", next);

        const restartButton = document.getElementById("restart-button");
        restartButton.addEventListener("click", restart);

    }

    function makeImagesClickable(clickable = true) {
        if (clickable) {
            imageContainers.forEach(container => {
                container.style.pointerEvents = "auto";
                container.addEventListener("click", handleImageClick);
                container.style.background = "none";
                container.classList.remove("clicked");
            });
        } else {
            imageContainers.forEach(imageContainer => imageContainer.style.pointerEvents = "none");
            imageContainers.forEach(imageContainer => imageContainer.removeEventListener("click", handleImageClick));
        }
    }

    function next() {
        currentQuestionIndex++;
        currentClickCount = 0;


        if (currentQuestionIndex < numQuestions) {
            if (gameMode === "play") {
                showUserFeedback(false);
                showPatchMatchCorrect(false);
                makeImagesClickable(true);
                feedbackText.style.color = "white";
                feedbackText.textContent = "Make a guess next time!"
            } else {
                showPatchMatchCorrect(true);
            }

            loadQuestion(currentQuestionIndex);
        } else {
            getResults();
        }
    }

    function getResults() {
        const nextButton = document.getElementById("next-button");
        if (nextButton) {
            nextButton.remove();
        }

        if (gameMode === "play") {
            if (correctAnswers > modelCorrectAnswers) {
                alert(`You matched ${correctAnswers} out of ${numQuestions} right and won against PatchMatchNN!`);
                feedbackText.style.color = "white";
                feedbackText.textContent = "Play again?"
            } else {
                if (correctAnswers === modelCorrectAnswers) {
                    alert(`You matched ${correctAnswers} out of ${numQuestions} right and tied with PatchMatchNN!`);
                    feedbackText.style.color = "white";
                    feedbackText.textContent = "Better luck next time!"

                } else {
                    alert(`You matched ${correctAnswers} out of ${numQuestions} right and lost to PatchMatchNN!`);
                    feedbackText.style.color = "white";
                    feedbackText.textContent = "Better luck next time!"
                }
            }
        }

    }
})



