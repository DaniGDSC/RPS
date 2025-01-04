import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import json
import pandas as pd

# Load or initialize Markov Chain data
try:
    with open("markov_data.json", "r") as file:
        data = json.load(file)
except FileNotFoundError:
    data = {
        "transitions": {str(i): {str(j): 1 for j in range(1, 4)} for i in range(1, 4)},
        "totals": {str(i): 3 for i in range(1, 4)}
    }

# Load game history
try:
    history = pd.read_csv("game_history.csv").to_dict(orient="records")
    round_num = len(history) + 1  # Continue from the last round
except FileNotFoundError:
    history = []
    round_num = 1

# Functions for Markov Chain
def update_transition_data(prev_move, current_move, data):
    data["transitions"][prev_move][current_move] += 1
    data["totals"][prev_move] += 1
    # Save Markov Chain data immediately
    with open("markov_data.json", "w") as file:
        json.dump(data, file)
    return data

def predict_next_move(current_move, data):
    probabilities = {
        move: count / data["totals"][current_move]
        for move, count in data["transitions"][current_move].items()
    }
    predicted_move = max(probabilities, key=probabilities.get)
    return (int(predicted_move) % 3) + 1  # Return the winning move

# Initialize variables
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
previousMove = None

# Functions for handling history
def update_history(history, round_num, player_move, ai_move, winner):
    history.append({
        "round": round_num,
        "player_move": player_move,
        "ai_move": ai_move,
        "winner": winner
    })
    # Save history to CSV immediately
    save_history_to_csv(history)

def save_history_to_csv(history, filename="game_history.csv"):
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)

def display_history(history):
    df = pd.DataFrame(history)
    print(df)

while True:
    imgBG = cv2.imread(r"C:\Users\LOAN\Downloads\RPS\Resources\BG.png")
    success, img = cap.read()

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    # Find Hands
    hands, img = detector.findHands(imgScaled)  # with draw

    if startGame:
        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if hands:
                    playerMove = None
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    elif fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    elif fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3

                    # Predict AI's move using Markov Chain
                    if previousMove is not None:
                        randomNumber = predict_next_move(str(previousMove), data)
                    else:
                        randomNumber = random.randint(1, 3)

                    # Update Markov Chain data
                    if playerMove is not None and previousMove is not None:
                        data = update_transition_data(str(previousMove), str(playerMove), data)
                    previousMove = playerMove

                    # Load AI's move image
                    imgAI = cv2.imread(f'C:/Users/LOAN/Downloads/RPS/Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                    # Determine winner
                    winner = "Draw"
                    if (playerMove == 1 and randomNumber == 3) or \
                       (playerMove == 2 and randomNumber == 1) or \
                       (playerMove == 3 and randomNumber == 2):
                        winner = "Player"
                        scores[1] += 1
                    elif (playerMove == 3 and randomNumber == 1) or \
                         (playerMove == 1 and randomNumber == 2) or \
                         (playerMove == 2 and randomNumber == 3):
                        winner = "AI"
                        scores[0] += 1

                    # Update history
                    update_history(history, round_num, playerMove, randomNumber, winner)
                    round_num += 1

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("BG", imgBG)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('q'):
        break

# Release resources and display history
cap.release()
cv2.destroyAllWindows()
display_history(history)
