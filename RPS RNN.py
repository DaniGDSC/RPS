import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

# Kiểm tra và tạo thư mục Resources nếu chưa tồn tại
RESOURCES_PATH = "Resources"
if not os.path.exists(RESOURCES_PATH):
    os.makedirs(RESOURCES_PATH)

# Kiểm tra và tạo file RNN_history.csv nếu chưa tồn tại
if not os.path.exists("RNN_history.csv"):
    df = pd.DataFrame(columns=["round", "player_move", "ai_move", "winner"])
    df.to_csv("RNN_history.csv", index=False)
    print("Created RNN_history.csv")

# Kiểm tra model RNN
try:
    model = load_model('rps_rnn.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using random predictions instead")
    model = None

# Constants
sequence_length = 5
move_history = []

# Load game history
def load_game_history():
    try:
        history = pd.read_csv("RNN_history.csv").to_dict(orient="records")
        round_num = len(history) + 1
    except FileNotFoundError:
        history = []
        round_num = 1
    return history, round_num

history, round_num = load_game_history()

def prepare_rnn_input(move_history):
    """Chuẩn bị input cho model RNN"""
    # Chỉ lấy sequence_length moves gần nhất
    if len(move_history) > sequence_length:
        move_history = move_history[-sequence_length:]
        
    if len(move_history) < sequence_length:
        return None
        
    # Chuyển đổi thành one-hot encoding
    input_data = np.zeros((1, sequence_length, 3))
    
    try:
        for i, move in enumerate(move_history):
            if move is not None and 1 <= move <= 3:
                input_data[0, i, move-1] = 1
    except Exception as e:
        print(f"Error preparing input: {e}")
        return None
            
    return input_data

def predict_next_move(move_history):
    """Dự đoán nước đi tiếp theo dựa trên model RNN"""
    if model is None:
        return random.randint(1, 3)
        
    try:
        input_data = prepare_rnn_input(move_history)
        
        if input_data is None:
            return random.randint(1, 3)
            
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_move = np.argmax(prediction) + 1
        
        # Trả về nước đi thắng predicted_move
        winning_moves = {1: 2, 2: 3, 3: 1}
        return winning_moves[predicted_move]
        
    except Exception as e:
        print(f"Error predicting move: {e}")
        return random.randint(1, 3)

def update_history(history, round_num, player_move, ai_move, winner):
    history.append({
        "round": round_num,
        "player_move": player_move,
        "ai_move": ai_move,
        "winner": winner
    })
    save_history_to_csv(history)

def save_history_to_csv(history, filename="RNN_history.csv"):
    df = pd.DataFrame(history)
    df.to_csv(filename, index=False)

def display_history(history):
    df = pd.DataFrame(history)
    print("\nGame History:")
    print(df)

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Game variables
timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
initialTime = 0
imgAI = None

# Main game loop
while True:
    try:
        # Load background image
        imgBG = cv2.imread(os.path.join(RESOURCES_PATH, "BG.png"))
        if imgBG is None:
            raise Exception("Could not load background image")

        # Read camera
        success, img = cap.read()
        if not success:
            raise Exception("Failed to capture camera frame")

        imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
        imgScaled = imgScaled[:, 80:480]

        # Find Hands
        hands, img = detector.findHands(imgScaled)

        if startGame:
            if stateResult is False:
                timer = time.time() - initialTime
                cv2.putText(imgBG, str(int(timer)), (605, 435), 
                          cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

                if timer > 3:
                    stateResult = True
                    timer = 0

                    if hands:
                        playerMove = None
                        hand = hands[0]
                        fingers = detector.fingersUp(hand)
                        
                        # Interpret hand gesture
                        if fingers == [0, 0, 0, 0, 0]:
                            playerMove = 1  # Rock
                        elif fingers == [1, 1, 1, 1, 1]:
                            playerMove = 2  # Paper
                        elif fingers == [0, 1, 1, 0, 0]:
                            playerMove = 3  # Scissors

                        # Get AI move
                        if playerMove is not None:
                            # Giới hạn độ dài của move_history
                            if len(move_history) >= sequence_length * 2:
                                move_history = move_history[-sequence_length:]
                            
                            move_history.append(playerMove)
                            randomNumber = predict_next_move(move_history)
                        else:
                            randomNumber = random.randint(1, 3)

                        # Load AI move image
                        imgAI = cv2.imread(os.path.join(RESOURCES_PATH, f"{randomNumber}.png"), 
                                         cv2.IMREAD_UNCHANGED)
                        if imgAI is not None:
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

        # Show game window
        imgBG[234:654, 795:1195] = imgScaled

        if stateResult and imgAI is not None:
            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

        # Show scores
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
            
    except Exception as e:
        print(f"Error in main loop: {e}")
        continue

cap.release()
cv2.destroyAllWindows()
display_history(history)
