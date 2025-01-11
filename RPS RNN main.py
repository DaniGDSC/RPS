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
    if len(move_history) > sequence_length:
        move_history = move_history[-sequence_length:]
    if len(move_history) < sequence_length:
        return None
    input_data = np.zeros((1, sequence_length, 3))
    for i, move in enumerate(move_history):
        if move is not None and 1 <= move <= 3:
            input_data[0, i, move - 1] = 1
    return input_data

def predict_next_move(move_history):
    if model is None:
        return random.randint(1, 3)
    try:
        input_data = prepare_rnn_input(move_history)
        if input_data is None:
            return random.randint(1, 3)
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_move = np.argmax(prediction) + 1
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

# Button properties
button_width = 150
button_height = 40
screen_width = 1195
buttons_y = 650

# Button definitions
start_button = {'x': screen_width // 2 - button_width * 1.5, 'y': buttons_y, 'w': button_width, 'h': button_height}
stop_button = {'x': screen_width // 2 - button_width // 2, 'y': buttons_y, 'w': button_width, 'h': button_height}
history_button = {'x': screen_width // 2 + button_width // 2, 'y': buttons_y, 'w': button_width, 'h': button_height}
exit_button = {'x': screen_width // 2 + button_width * 1.5, 'y': buttons_y, 'w': button_width, 'h': button_height}

showing_history = False
mouse_callback_set = False
running = True

def mouse_callback(event, x, y, flags, param):
    global startGame, showing_history, running

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pos = (x, y)
        if is_mouse_over_button(mouse_pos, start_button):
            startGame = True
            global initialTime, stateResult
            initialTime = time.time()
            stateResult = False
        elif is_mouse_over_button(mouse_pos, stop_button):
            startGame = False
        elif is_mouse_over_button(mouse_pos, history_button):
            showing_history = not showing_history
        elif is_mouse_over_button(mouse_pos, exit_button):
            running = False

def is_mouse_over_button(mouse_pos, button):
    return (button['x'] <= mouse_pos[0] <= button['x'] + button['w'] and
            button['y'] <= mouse_pos[1] <= button['y'] + button['h'])

def draw_button(img, button, text, color=(255, 255, 255), text_color=(0, 0, 0)):
    cv2.rectangle(img,
                  (int(button['x']), int(button['y'])),
                  (int(button['x'] + button['w']), int(button['y'] + button['h'])),
                  color,
                  cv2.FILLED)
    cv2.rectangle(img,
                  (int(button['x']), int(button['y'])),
                  (int(button['x'] + button['w']), int(button['y'] + button['h'])),
                  (0, 0, 0),
                  2)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.5
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = int(button['x'] + (button['w'] - text_size[0]) // 2)
    text_y = int(button['y'] + (button['h'] + text_size[1]) // 2)
    cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

# Main game loop
while running:
    try:
        if not mouse_callback_set:
            cv2.namedWindow('BG')
            cv2.setMouseCallback('BG', mouse_callback)
            mouse_callback_set = True

        imgBG = cv2.imread(os.path.join(RESOURCES_PATH, "BG.png"))
        if imgBG is None:
            raise Exception("Could not load background image")

        success, img = cap.read()
        if not success:
            raise Exception("Failed to capture camera frame")

        imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
        imgScaled = imgScaled[:, 80:480]

        hands, img = detector.findHands(imgScaled)

        if startGame:
            if not stateResult:
                timer = time.time() - initialTime
                cv2.putText(imgBG, str(int(timer)), (605, 435),
                            cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)
                if timer > 3:
                    stateResult = True
                    timer = 0
                    playerMove = None
                    if hands:
                        hand = hands[0]
                        fingers = detector.fingersUp(hand)
                        if fingers == [0, 0, 0, 0, 0]:
                            playerMove = 1  # Rock
                        elif fingers == [1, 1, 1, 1, 1]:
                            playerMove = 2  # Paper
                        elif fingers == [0, 1, 1, 0, 0]:
                            playerMove = 3  # Scissors

                        if playerMove is not None:
                            if len(move_history) >= sequence_length * 2:
                                move_history = move_history[-sequence_length:]
                            move_history.append(playerMove)
                            randomNumber = predict_next_move(move_history)
                        else:
                            randomNumber = random.randint(1, 3)

                        imgAI = cv2.imread(os.path.join(RESOURCES_PATH, f"{randomNumber}.png"),
                                           cv2.IMREAD_UNCHANGED)
                        if imgAI is not None:
                            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

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

                        update_history(history, round_num, playerMove, randomNumber, winner)
                        round_num += 1

        imgBG[234:654, 795:1195] = imgScaled

        if stateResult and imgAI is not None:
            imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

        cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
        cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

        # Draw buttons
        draw_button(imgBG, start_button, "Start", (0, 255, 0) if not startGame else (200, 200, 200))
        draw_button(imgBG, stop_button, "Stop", (255, 0, 0) if startGame else (200, 200, 200))
        draw_button(imgBG, history_button, "History", (0, 255, 255) if showing_history else (200, 200, 200))
        draw_button(imgBG, exit_button, "Exit", (255, 0, 255))

        if showing_history:
            overlay = imgBG.copy()
            cv2.rectangle(overlay, (300, 100), (900, 600), (255, 255, 255), cv2.FILLED)
            cv2.rectangle(overlay, (300, 100), (900, 600), (0, 0, 0), 2)

            y_offset = 150
            cv2.putText(overlay, "Game History (Last 15 rounds):", (320, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)

            # Lấy 15 vòng cuối cùng từ lịch sử
            recent_history = history[-15:] if len(history) > 15 else history

            for game in recent_history:
                text = f"Round {game['round']}: Player {game['player_move']} vs AI {game['ai_move']} - {game['winner']}"
                cv2.putText(overlay, text, (320, y_offset), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1)
                y_offset += 30

            alpha = 0.7
            imgBG = cv2.addWeighted(overlay, alpha, imgBG, 1 - alpha, 0)


        cv2.imshow("BG", imgBG)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    except Exception as e:
        print(f"An error occurred: {e}")
        break

cap.release()
cv2.destroyAllWindows()
