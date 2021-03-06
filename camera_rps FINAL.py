import cv2
from keras.models import load_model 
import numpy as np 
import random 
import time 

def detectClick(event, x, y, flags, param):
    global gameIntro
    global countdown 
    global choiceFlag
    if event == cv2.EVENT_LBUTTONDOWN and gamePlay == False:
        gameIntro = True 
        countdown = currentTime + 5 
    if event == cv2.EVENT_RBUTTONDOWN and gamePlay == True:
        gameIntro = True 
        choiceFlag = False 
        countdown = currentTime + 5 
        gameIntroGenerate()

def createPlayerChoice():
    global solidPlayerChoice 
    # Choose the best probable value from a buffer of 12 previous lines
    maxValue = np.argmax(prediction)
    playerChosen = (playerChoice[maxValue])

    if len(bufferPlayerChoice) < 12:
        bufferPlayerChoice.append(playerChosen) 
    else:
        bufferPlayerChoice.pop(0)
        bufferPlayerChoice.append(playerChosen)

# Rock is a bit fragile therefore best option if 2 or more in buffer 
    if bufferPlayerChoice.count("Rock") >= 2:
        solidPlayerChoice = "Rock"
    else:
        solidPlayerChoice = max(bufferPlayerChoice, key = bufferPlayerChoice.count)

def winnerEval(comp, player):
    global cWins 
    global pWins 
    global round 
    global updated 
    if comp == player:
        winner = "Draw"
    elif comp == "Rock" and player == "Scissors":
        winner = "C"
    elif comp == "Paper" and player == "Rock":
        winner = "C"
    elif comp == "Scissors" and player == "Paper":
        winner = "C"
    elif player == "-Waiting-":
        winner = "Error"
    else:
        winner = "P"
    if updated == False:
        if winner == "C":
            cWins += 1 
            round += 1
            updated = True
        elif winner == "P":
            pWins += 1
            round += 1 
            updated = True 
        
    print(round)
    return winner 

def placeholderGenerate():
    createPlayerChoice()
    # Make screen have boxes and look good 
    # White box background 
    cv2.rectangle(frame, (10,430), (720,350), (255,255,255), -1)
    # Player label
    cv2.putText(frame, "Player", (70,389), 1, 2, (0,0,0))
    # Computer label
    cv2.putText(frame, "Computer", (510,389), 1, 2, (0,0,0)) 
    # vs separator 
    cv2.putText(frame, "V", (340,420), 2, 3, (0,0,0))
    # Create the tect of the interpreted choice from the buffer
    cv2.putText(frame, solidPlayerChoice, (20,420), 0, 1, (0,0,0))
    # Green button 
    cv2.rectangle(frame, (175,80), (275,120), (20,120,20), -1)
    # Begin text 
    cv2.putText(frame, "Click", (60,110), 3, 1, (0,0,0))
    
def randCompChoice():
    global compChoice
    global choiceFlag 
    compChoice = random.choice(playerChoice[0:3])

def gameIntroGenerate():
    createPlayerChoice()
    global choiceFlag 
    global gameIntro 
    global gamePlay
    global updated 
    global playerChoice 
    updated = False
    if choiceFlag == False:
        randCompChoice()
        choiceFlag = True 
    if gameIntro:
        cv2.rectangle(frame, (10,430), (720,350), (0,255,255), -1)
        # Player label 
        cv2.putText(frame, "Player", (70,380), 1, 2, (0,0,0))
        # Computer label 
        cv2.putText(frame, "Computer", (510, 380), 1, 2, (0,0,0))
        # vs separator 
        cv2.putText(frame, str(int(countdown-currentTime)), (190,420), 2, 3, (0,0,0))
        # Create the text of the computer's random choice 
        cv2.putText(frame, solidPlayerChoice, (20,420), 0, 1, (0,0,0))
        # Create the text of the computer's random choice 
        cv2.putText(frame, random.choice(playerChoice[0:3]), (280,420), 0, 1, (0,0,0))
        if int(countdown-currentTime) <= 0:
            gameIntro = False
            gamePlay = True 
            gamePlayGenerate()

def gamePlayGenerate():
    global first 
    global gamePlay 
    global gameIntro 
    print("PLAYING!!!")
    # White box background 
    cv2.rectangle(frame, (10,430), (720,350), (200,200,200), -1)
    # Player label 
    cv2.putText(frame, "Player", (70,380), 1, 2, (0,0,0))
    # Computer label 
    cv2.putText(frame, "Computer", (510,380), 1, 2, (0,0,0))
    # vs separator 
    cv2.putText(frame, "V", (340,420), 2, 3, (0,0,0))
    # Create the tetxt of the interpretted choice from the buffer 
    cv2.putText(frame, solidPlayerChoice, (20,420), 4, 1, (0,0,0))
    # Create the text of the computer's random choice 
    cv2.putText(frame, compChoice, (280,420), 0, 1, (0,0,0))
    # Green button 
    cv2.rectangle(frame, (50,80), (402,120), (20,170,20), -1)
    # Check to see if there's an overall winner 

    def gameOver(winner):
        cv2.rectangle(frame, (50,80), (402,160), (180,170,150), -1)
        cv2.putText(frame, "GAME OVER", (135,110), 3, 1, (0,0,0))
        cv2.putText(frame, winner + " WINS!", (110,150), 3, 1, (0,0,0))
     
    if cWins == 3:
        gamePlay = False
        gameIntro = True 
        gameOver("Computer")
    elif pWins == 3:
        gameIntro = False 
        gamePlay = True 
        gameOver("Player")
      
    else:
        # Actually decide on a winner 
        if winnerEval(compChoice, solidPlayerChoice) == "C":
            cv2.putText(frame, "CPU Wins!", (140,110), 3, 1, (0,0,0))
        elif winnerEval(compChoice, solidPlayerChoice) == "P":
            cv2.putText(frame, "Player Wins!", (110, 110), 3, 1, (0,0,0))
        elif winnerEval(compChoice, solidPlayerChoice) == "Draw":
            cv2.putText(frame, "Draw", (165,110), 3, 1, (0,0,0))
        else:
            cv2.putText(frame, "Too Slow", (150,110), 3, 1, (0,0,0))

    
  

    # Declare variables
model = load_model('keras_model.h5')
capture = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
playerChoice = ["Rock", "Paper", "Scissors", "-Waiting-", "Play Game"]
size = (224, 224)


bufferPlayerChoice = []
solidPlayerChoice = ""
gameIntro = False 
gamePlay = False 
countdown = 3 
compChoice = "EMPTY"
choiceFlag = False 
first = True 
pWins = 0 
cWins = 0 
round = 1 
updated = False 
theEnd = False 


while True: 
    # Not sure what ret is??
    ret, frame = capture.read()
    # Crop the frame to match teachablemachine's cropped area
    if frame is None:
        print("No Image")
        continue # Move to top of while loop without further processing of this iteration 
    else: 
        frame = frame[32:, 118:]

    # Resize the frame to 224 x 224 
    resized_frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalise the image 
    data[0] = normalized_image
    prediction = model.predict(data)
    print(prediction)
    
    # Generate persistent items 
    cv2.rectangle(frame, (10,10), (720, 50), (225,225,0), -1)
    message = "Round: " +str(round) + "     " + "Score: " + str(pWins) + ":" +str(cWins)
    cv2.putText(frame, message, (20,40), 1, 2, (0,0,0))

    # Output the best choice to the shell 
    print(solidPlayerChoice)
    print(compChoice)
    print(time.time())
    currentTime = time.time()

    # Generates appropiate display overlay 
    if theEnd == True: 
        pass
    elif gameIntro == True: 
        gameIntroGenerate()
    elif gamePlay == True and gameIntro == False:
        gamePlayGenerate()
    else: 
        placeholderGenerate()

    # Output of the frame window with nice title 
    #cv2.namedWindow("Rock Paper Scissors")#
    cv2.imshow("Rock Paper Scissors", frame)

    # Click mouse anywhere to begin 
    cv2.setMouseCallback("Rock Paper Scissors", detectClick)

    # Press q to close window 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break





